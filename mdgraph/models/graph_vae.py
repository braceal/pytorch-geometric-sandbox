import torch
from torch import nn
import torch.optim
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, GAE
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_geometric.utils.repeat import repeat

import argparse
from tqdm import tqdm
from pathlib import Path
from mdgraph.data.dataset import ContactMapDataset
from torch_geometric.data import DataLoader

MAX_LOGSTD = 10


def kl_loss(mu: torch.Tensor, logstd: torch.Tensor):
    r"""Computes the KL loss, either for the passed arguments :obj:`mu`
    and :obj:`logstd`.
    Args:
        mu (Tensor): The latent space for :math:`\mu`.
        logstd (Tensor): The latent space for :math:`\log\sigma`.
    """
    logstd = logstd.clamp(max=MAX_LOGSTD)
    return -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1)
    )


def reparametrize(mu: torch.Tensor, logstd: torch.Tensor, training: bool = False):
    if training:
        return mu + torch.randn_like(logstd) * torch.exp(logstd)
    else:
        return mu


class UpPool(nn.Module):
    def __init__(self):
        super(UpPool, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, res, perm):
        up = torch.zeros_like(res)
        up[perm] = x
        return up


class LinearEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGraphEncoder(nn.Module):
    """Acts on NxD node embedding matrix."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 1,
        pool_ratios: float = 0.5,
        act=F.relu,
    ):
        super(VariationalGraphEncoder, self).__init__()
        assert depth >= 1
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, hidden_channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(hidden_channels, self.pool_ratios[i]))
            self.down_convs.append(
                GCNConv(hidden_channels, hidden_channels, improved=True)
            )

        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.conv_mu.reset_parameters()
        self.conv_logstd.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        x, edge_index, xs, edge_indices, edge_weights, perms = self.down_sample(
            x, edge_index, batch
        )
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        z = reparametrize(mu, logstd, training=self.training)
        return z, mu, logstd, xs, edge_indices, edge_weights, perms

    def down_sample(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(
                edge_index, edge_weight, x.size(0)
            )
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch
            )

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            # Collect data for skip connections
            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        return x, edge_index, xs, edge_indices, edge_weights, perms

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes)
        edge_index, edge_weight = spspmm(
            edge_index,
            edge_weight,
            edge_index,
            edge_weight,
            num_nodes,
            num_nodes,
            num_nodes,
        )
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight


class VariationalGraphDecoder(nn.Module):
    """Acts on NxD node embedding matrix."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 1,
        sum_res: bool = True,
        act=F.relu,
    ):
        super(VariationalGraphDecoder, self).__init__()
        assert depth >= 1

        self.depth = depth
        self.sum_res = sum_res
        self.act = act

        self.up_convs = nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, hidden_channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, xs, edge_indices, edge_weights, perms):
        x = self.up_sample(x, xs, edge_indices, edge_weights, perms)
        return x

    def up_sample(self, x, xs, edge_indices, edge_weights, perms):
        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x


parser = argparse.ArgumentParser()
parser.add_argument("--linear", action="store_true")
parser.add_argument("--epochs", type=int, default=400)
args = parser.parse_args()

# Architecture Hyperparameters
num_features = 5
node_out_channels = 10
graph_out_channels = 10
hidden_channels = 10
depth = 2
pool_ratios = 0.5
act = F.relu
sum_res = True

path = Path(__file__).parent / "../../test/data/BBA-subset-100.h5"
dataset = ContactMapDataset(path, "contact_map", ["rmsd"], num_features)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Select node AE
if args.linear:
    node_ae = GAE(LinearEncoder(num_features, node_out_channels))
else:
    node_ae = GAE(GCNEncoder(num_features, node_out_channels))

# Select graph AE
encoder = VariationalGraphEncoder(
    node_out_channels,
    hidden_channels,
    graph_out_channels,
    depth,
    pool_ratios,
    act,
)
decoder = VariationalGraphDecoder(
    graph_out_channels,
    hidden_channels,
    node_out_channels,
    depth,
    sum_res,
    act,
)

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_ae = node_ae.to(device)
encoder, decoder = encoder.to(device), decoder.to(device)

# Optimizer
optimizer = torch.optim.Adam(
    set(node_ae.parameters()) | set(encoder.parameters()) | set(decoder.parameters()),
    lr=0.01,
)


def train():
    node_ae.train()
    encoder.train()
    decoder.train()

    train_loss = 0.0
    for i, sample in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        data = sample["X"]
        data = data.to(device)

        # Get NxD node embedding matrix
        node_z = node_ae.encode(data.x, data.edge_index)
        print("node_z shape:", node_z.shape)
        # Get graph embedding
        graph_z, mu, logstd, xs, edge_indices, edge_weights, perms = encoder(
            node_z, data.edge_index
        )
        print("graph_z shape:", graph_z.shape)
        # Reconstruct node embedding matrix
        node_z_recon = decoder(graph_z, xs, edge_indices, edge_weights, perms)
        print("node_z_recon shape:", node_z_recon.shape)

        recon_loss = node_ae.recon_loss(node_z_recon, data.edge_index)
        print("recon_loss:", recon_loss)
        kld_loss = (1 / data.num_nodes) * kl_loss(mu, logstd)
        print("kld loss:", kld_loss)
        print("data.num_nodes:", data.num_nodes)
        loss = recon_loss + kld_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss


for epoch in range(1, args.epochs + 1):
    loss = train()
    print(f"Epoch: {epoch:03d}\tLoss: {loss}")
