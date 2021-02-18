import torch
from torch import nn
import torch.optim
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, GINConv, GAE, VGAE
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_geometric.utils.repeat import repeat

import time
import argparse
import numpy as np

# from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from mdgraph.data.dataset import ContactMapDataset
from torch_geometric.data import DataLoader
from mdgraph.utils import tsne_validation

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


class GINLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINLayer, self).__init__()
        seq = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.conv = GINConv(seq)
        self.bn = nn.BatchNorm1d(out_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()
        # TODO: reset bn?

    def forward(self, x, edge_index):
        x = F.relu(self.conv(x, edge_index))
        x = self.bn(x)
        return x


def generate_conv(in_channels: int, out_channels: int, use_gin: bool):
    if use_gin:
        return GINLayer(in_channels, out_channels)
    else:
        return GCNConv(in_channels, out_channels)


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


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


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
        variational: bool = True,
        use_gin: bool = False,
    ):
        super(VariationalGraphEncoder, self).__init__()
        assert depth >= 1
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.variational = variational

        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, hidden_channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(hidden_channels, self.pool_ratios[i]))
            self.down_convs.append(
                GCNConv(hidden_channels, hidden_channels, improved=True)
            )
        self.conv = generate_conv(hidden_channels, hidden_channels // 2, use_gin)
        self.conv_mu = generate_conv(hidden_channels // 2, out_channels, use_gin)
        if self.variational:
            self.conv_logstd = generate_conv(hidden_channels, out_channels, use_gin)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.conv.reset_parameters()
        self.conv_mu.reset_parameters()
        if self.variational:
            self.conv_logstd.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        x, edge_index, xs, edge_indices, edge_weights, perms = self.down_sample(
            x, edge_index, batch
        )
        x = self.conv(x, edge_index)
        mu = self.conv_mu(x, edge_index)
        if self.variational:
            logstd = self.conv_logstd(x, edge_index)
            z = reparametrize(mu, logstd, training=self.training)
        else:
            z = mu
            logstd = None
        return z, mu, logstd, edge_index, xs, edge_indices, edge_weights, perms

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
        use_gin: bool = False,
    ):
        super(VariationalGraphDecoder, self).__init__()
        assert depth >= 1

        self.depth = depth
        self.sum_res = sum_res
        self.act = act

        # Before GIN, it was using GCNConv with improved=True
        self.projection_conv1 = generate_conv(
            in_channels, hidden_channels // 2, use_gin
        )
        self.projection_conv2 = generate_conv(
            hidden_channels // 2, hidden_channels, use_gin
        )
        self.up_convs = nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(
                GCNConv(hidden_channels, hidden_channels, improved=True)
            )
        self.up_convs.append(GCNConv(hidden_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        self.projection_conv1.reset_parameters()
        self.projection_conv2.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, xs, edge_indices, edge_weights, perms):
        x = self.projection_conv1(x, edge_index)
        x = self.projection_conv2(x, edge_index)
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
            # print("up.shape:", up.shape)
            # print("x.shape:", x.shape)
            up[perm] = x
            # x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
            x = up
            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x


parser = argparse.ArgumentParser()
parser.add_argument("--linear", action="store_true")
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--name", type=str)
args = parser.parse_args()

# Architecture Hyperparameters
num_features = 13
node_out_channels = 10
graph_out_channels = 1
hidden_channels = 10
depth = 2
pool_ratios = 0.5
act = F.relu
sum_res = True
variational = False
node_recon_loss_weight = 10.0
use_gin = True

path = Path(__file__).parent / "../../test/data/BBA-subset-100.h5"
node_feature_path = (
    Path(__file__).parent / "../../test/data/onehot_bba_amino_acid_labels.npy"
)
dataset = ContactMapDataset(
    path, "contact_map", ["rmsd"], node_feature_path=node_feature_path
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Select node AE
if args.linear:
    node_ae = GAE(GCNEncoder(num_features, node_out_channels))
else:
    node_ae = VGAE(VariationalGCNEncoder(num_features, node_out_channels))

# Select graph AE
encoder = VariationalGraphEncoder(
    node_out_channels,
    hidden_channels,
    graph_out_channels,
    depth,
    pool_ratios,
    act,
    variational,
    use_gin,
)
decoder = VariationalGraphDecoder(
    graph_out_channels,
    hidden_channels,
    node_out_channels,
    depth,
    sum_res,
    act,
    use_gin,
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

node_emb_recon_criterion = nn.MSELoss()


def train():
    node_ae.train()
    encoder.train()
    decoder.train()

    train_loss = 0.0
    for i, sample in enumerate(loader):
        start = time.time()
        optimizer.zero_grad()
        data = sample["X"]
        data = data.to(device)

        # Get NxD node embedding matrix
        node_z = node_ae.encode(data.x, data.edge_index)
        # print("node_z shape:", node_z.shape)
        # Get graph embedding
        (
            graph_z,
            mu,
            logstd,
            edge_index,
            xs,
            edge_indices,
            edge_weights,
            perms,
        ) = encoder(node_z, data.edge_index)
        assert logstd is None
        # print("graph_z shape:", graph_z.shape)
        # Reconstruct node embedding matrix
        node_z_recon = decoder(
            graph_z, edge_index, xs, edge_indices, edge_weights, perms
        )
        # print("node_z_recon shape:", node_z_recon.shape)

        # Adjacency matrix reconstruction
        loss = node_ae.recon_loss(node_z_recon, data.edge_index)
        # print("node_ae rec loss:",loss)
        # Node embedding reconstruction
        rec_loss = node_recon_loss_weight * node_emb_recon_criterion(
            node_z, node_z_recon
        )
        # print("rec_loss:",rec_loss)
        loss += rec_loss
        if not args.linear:
            loss += (1 / data.num_nodes) * node_ae.kl_loss()
        # print("recon_loss:", recon_loss)
        if variational:
            assert False
            kld_loss = kl_loss(mu, logstd)
            loss += kld_loss
        # print("kld loss:", kld_loss)
        # print("data.num_nodes:", data.num_nodes)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if i % 100 == 0:
            print(
                f"Training {i}/{len(loader)}. Loss: {train_loss / (i + 1)} Batch time: {time.time() - start}"
            )

    train_loss /= len(loader)

    return train_loss


def validate_with_rmsd():
    node_ae.eval()
    encoder.eval()
    decoder.eval()

    output = defaultdict(list)
    with torch.no_grad():
        for sample in loader:
            data = sample["X"]
            data = data.to(device)

            node_z = node_ae.encode(data.x, data.edge_index)
            (
                graph_z,
                mu,
                logstd,
                edge_index,
                xs,
                edge_indices,
                edge_weights,
                perms,
            ) = encoder(node_z, data.edge_index)
            node_z_recon = decoder(
                graph_z, edge_index, xs, edge_indices, edge_weights, perms
            )

            # Collect embeddings for plot
            node_emb = node_z.detach().cpu().numpy()
            node_recon_emb = node_z_recon.detach().cpu().numpy()
            graph_emb = graph_z.detach().cpu().numpy()
            output["graph_embeddings"].append(graph_emb)
            output["node_embeddings"].append(node_emb)
            output["node_labels"].append(data.y.detach().cpu().numpy())
            output["node_recon_embeddings"].append(node_recon_emb)
            output["rmsd"].append(sample["rmsd"].detach().cpu().numpy())

    output = {key: np.array(val).squeeze() for key, val in output.items()}

    shape = output["node_embeddings"].shape
    for name in ["node_embeddings", "node_recon_embeddings"]:
        output[name] = output[name].reshape(shape[0] * shape[1], -1)

    output["node_labels"] = output["node_labels"].flatten()

    return output


def validate(epoch: int):

    output = validate_with_rmsd()

    print("graph_embeddings.shape:", output["graph_embeddings"].shape)
    print("node_embeddings.shape:", output["node_embeddings"].shape)
    print("rmsd.shape:", output["rmsd"].shape)

    # Paint graph embeddings
    random_sample = np.random.choice(
        len(output["graph_embeddings"]), 8000, replace=False
    )
    tsne_validation(
        embeddings=output["graph_embeddings"][random_sample],
        paint=output["rmsd"][random_sample],
        paint_name="rmsd",
        plot_dir=Path("./test_plots"),
        plot_name=f"epoch-{epoch}-graph_embeddings",
    )

    # Paint node embeddings
    random_sample = np.random.choice(
        len(output["node_embeddings"]), 8000, replace=False
    )
    tsne_validation(
        embeddings=output["node_embeddings"][random_sample],
        paint=output["node_labels"][random_sample],
        paint_name="node_labels",
        plot_dir=Path("./test_plots"),
        plot_name=f"epoch-{epoch}-node_embeddings",
    )

    # Paint node reconstruction embeddings
    random_sample = np.random.choice(
        len(output["node_recon_embeddings"]), 8000, replace=False
    )
    tsne_validation(
        embeddings=output["node_recon_embeddings"][random_sample],
        paint=output["node_labels"][random_sample],
        paint_name="node_labels",
        plot_dir=Path("./test_plots"),
        plot_name=f"epoch-{epoch}-node_recon_embeddings",
    )


for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    print(f"Epoch: {epoch:03d}\tLoss: {loss}\t Time: {time.time() - start}")
    start = time.time()
    validate(epoch)
    print(f"validation time: {time.time() - start}")
