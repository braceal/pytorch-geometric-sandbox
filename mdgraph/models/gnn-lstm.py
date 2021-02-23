import time
import argparse
from itertools import chain
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple
import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.data import DataLoader
from mdgraph.data.dataset import ContactMapDataset
from mdgraph.utils import tsne_validation

EPS = 1e-15
MAX_LOGSTD = 10

parser = argparse.ArgumentParser()
parser.add_argument(
    "--variational_node", action="store_true", help="Use variational node encoder"
)
parser.add_argument(
    "--use_node_z",
    action="store_true",
    help="Compute adjacency matrix reconstruction using "
    "node_encoder embeddings instead of LSTM decoder output",
)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument(
    "--split_pct",
    type=float,
    default=0.8,
    help="Percentage of data to use for training. The rest goes to validation.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=str(Path(__file__).parent / "../../test/data/BBA-subset-100.h5"),
)
args = parser.parse_args()


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


def recon_loss(decoder, z, pos_edge_index, neg_edge_index=None):
    r"""Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.
    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to train against.
        neg_edge_index (LongTensor, optional): The negative edges to train
            against. If not given, uses negative sampling to calculate
            negative edges. (default: :obj:`None`)
    """

    pos_loss = -torch.log(decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    # Do not include self-loops in negative samples
    pos_edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index, _ = add_self_loops(pos_edge_index)
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 - decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

    return pos_loss + neg_loss


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        z = reparametrize(mu, logstd, self.training)
        return z


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Parameters
        ----------
        input_size: int
            The number of expected features in the input x.
        hidden_size: int
            The number of features in the hidden state h.
        num_layers: int
            Number of recurrent layers. E.g., setting num_layers=2 would mean
            stacking two LSTMs together to form a stacked LSTM, with the second
            LSTM taking in outputs of the first LSTM and computing the final
            results. Default: 1
        bias: bool
            If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        dropout: float
            If non-zero, introduces a Dropout layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal
            to dropout. Default: 0
        bidirectional: bool
            If True, becomes a bidirectional LSTM. Default: False
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Hidden state logic: https://discuss.pytorch.org/t/lstm-hidden-state-logic/48101

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape BxNxD for B batches of N nodes
            by D node latent dimensions.
        """
        # batch_size = x.shape[0]
        # print("lstm encoder x.shape:", x.shape)
        _, (h_n, c_n) = self.lstm(x)  # output, (h_n, c_n)
        # print("enc h_n.shape:", h_n.shape)
        # print("enc c_n.shape:", c_n.shape)
        # print("enc output.shape:", output.shape)
        return h_n, c_n


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Parameters
        ----------
        input_size: int
            The number of expected features in the input x.
        hidden_size: int
            The number of features in the hidden state h.
        num_layers: int
            Number of recurrent layers. E.g., setting num_layers=2 would mean
            stacking two LSTMs together to form a stacked LSTM, with the second
            LSTM taking in outputs of the first LSTM and computing the final
            results. Default: 1
        bias: bool
            If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        dropout: float
            If non-zero, introduces a Dropout layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal
            to dropout. Default: 0
        bidirectional: bool
            If True, becomes a bidirectional LSTM. Default: False
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # TODO:
        # Analyze: Does it reconstruct links between AAs close to each other, how about far away?
        # Add link prediction task for long distance links (far away in AA seq)
        # Add multitask regression head for potential energy

        # Hidden state logic: https://discuss.pytorch.org/t/lstm-hidden-state-logic/48101
        # TODO: add gradient clipping: https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb

    def forward(self, x: torch.Tensor, h_n: torch.Tensor, c_n: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape BxNxD for B batches of N nodes
            by D node latent dimensions.
        """
        batch_size, seq_len, input_size = x.shape
        assert input_size == self.input_size

        outputs = torch.zeros_like(x)  # (batch_size, seq_len, input_size)
        # print("decoder outputs.shape:", outputs.shape)
        x_i = h_n.view(
            batch_size, self.num_layers * self.num_directions, self.hidden_size
        )

        # print("decoder x.shape:", x.shape)
        # print("decoder outputs.shape:", outputs.shape)
        for i in range(seq_len):
            # print("decoder x_i.shape:", x_i.shape)
            # print("decoder h_n.shape:", h_n.shape)
            # print("decoder c_n.shape:", c_n.shape)
            # print("decoder x_i.shape:", x_i.shape)
            output, (h_n, c_n) = self.lstm(x_i, (h_n, c_n))
            x_i = x[:, i].view(
                batch_size, self.num_layers * self.num_directions, input_size
            )
            # TODO: handle bidirectional output shape: (batch, seq_len==1, num_directions * hidden_size)
            # print("decoder x_i.shape:", x_i.shape)
            # print("decoder output.shape:", output.shape)
            # print("decoder output.sqeeuze().shape", output.squeeze().shape)
            outputs[:, i, :] = output.squeeze()

        return outputs


class LSTM_AE(nn.Module):
    r"""LSTM Autoencoder model from: https://arxiv.org/pdf/1502.04681.pdf"""

    def __init__(self, input_size: int, hidden_size: int, **kwargs):
        super().__init__()

        self.encoder = LSTMEncoder(input_size, hidden_size, **kwargs)
        self.decoder = LSTMDecoder(hidden_size, input_size, **kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("lstm ae: x.shape:", x.shape)
        h_n, c_n = self.encoder(x)
        # TODO: verify flip logic
        flipped_x = torch.flip(x, dims=(2,))
        decoded = self.decoder(flipped_x, h_n, c_n)
        decoded = torch.flip(decoded, dims=(2,))
        return h_n, decoded


# Parameters
out_channels = 10
num_features = 20
lstm_latent_dim = 10

# Data
dataset = ContactMapDataset(args.data_path, "contact_map", ["rmsd"])
lengths = [
    int(len(dataset) * args.split_pct),
    int(len(dataset) * round(1 - args.split_pct, 2)),
]
train_dataset, valid_dataset = random_split(dataset, lengths)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=True, drop_last=True)
num_nodes = dataset.num_nodes

# Models
if args.variational_node:
    node_encoder = VariationalGCNEncoder(num_features, out_channels)
else:
    node_encoder = GCNEncoder(num_features, out_channels)
node_decoder = InnerProductDecoder()
lstm_ae = LSTM_AE(out_channels, lstm_latent_dim)

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_encoder = node_encoder.to(device)
node_decoder = node_decoder.to(device)
lstm_ae = lstm_ae.to(device)

# Optimizer
optimizer = torch.optim.Adam(
    chain(node_encoder.parameters(), node_decoder.parameters(), lstm_ae.parameters()),
    lr=args.lr,
)

# Criterion
node_emb_recon_criterion = nn.MSELoss()


def train(epoch: int):
    node_encoder.train()
    node_decoder.train()
    lstm_ae.train()

    total_loss = 0.0
    for i, sample in enumerate(train_loader):
        start = time.time()
        optimizer.zero_grad()
        data = sample["data"]
        data = data.to(device)

        if args.variational_node:
            node_z, node_logstd = node_encoder(data.x, data.edge_index)
        else:
            node_z = node_encoder(data.x, data.edge_index)

        # print("node_z.shape:", node_z.shape)
        node_z = node_z.view(args.batch_size, num_nodes, out_channels)
        graph_z, node_z_recon = lstm_ae(node_z)

        # Reconstruction losses
        loss = recon_loss(
            node_decoder, node_z if args.use_node_z else node_z_recon, data.edge_index
        )
        loss += node_emb_recon_criterion(node_z, node_z_recon)

        # Variational losses
        if args.variational_node:
            loss += (1 / num_nodes) * kl_loss(node_z, node_logstd)

        # print(graph_emb.shape)
        # print(node_z_recon.shape)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(
            f"Training Epoch: {epoch} Batch: {i}/{len(train_loader)} "
            f"Loss: {total_loss / (i + 1)} Batch time: {time.time() - start}"
        )

    total_loss /= len(train_loader)

    return total_loss


def validate_with_rmsd():
    node_encoder.eval()
    node_decoder.eval()
    lstm_ae.eval()
    output = defaultdict(list)
    total_loss = 0.0
    with torch.no_grad():
        for sample in tqdm(valid_loader):
            data = sample["data"]
            data = data.to(device)

            if args.variational_node:
                node_z, node_logstd = node_encoder(data.x, data.edge_index)
            else:
                node_z = node_encoder(data.x, data.edge_index)

            node_z = node_z.view(args.batch_size, num_nodes, out_channels)
            graph_z, node_z_recon = lstm_ae(node_z)

            # Reconstruction losses
            loss = recon_loss(
                node_decoder,
                node_z if args.use_node_z else node_z_recon,
                data.edge_index,
            )
            loss += node_emb_recon_criterion(node_z, node_z_recon)

            # Variational losses
            if args.variational_node:
                loss += (1 / num_nodes) * kl_loss(node_z, node_logstd)

            total_loss += loss.item()

            # Collect embeddings for plot
            node_emb = (
                node_z.view(args.batch_size * num_nodes, out_channels)
                .detach()
                .cpu()
                .numpy()
            )
            graph_emb = graph_z.detach().cpu().numpy()
            output["graph_embeddings"].extend(graph_emb)
            output["node_embeddings"].append(node_emb)
            output["node_labels"].append(data.y.detach().cpu().numpy())
            output["rmsd"].extend(list(sample["rmsd"].detach().cpu().numpy()))

    output = {key: np.array(val).squeeze() for key, val in output.items()}

    shape = output["node_embeddings"].shape
    output["node_embeddings"] = output["node_embeddings"].reshape(
        shape[0] * shape[1], -1
    )
    output["graph_embeddings"] = output["graph_embeddings"].reshape(
        args.batch_size * len(valid_loader), lstm_latent_dim
    )
    output["node_labels"] = output["node_labels"].flatten()

    total_loss /= len(valid_loader)
    return output, total_loss


def validate(epoch: int):

    output, total_loss = validate_with_rmsd()

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

    return total_loss


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    valid_loss = validate(epoch)
    print(f"Epoch: {epoch:03d}\t Train Loss: {train_loss} Valid Loss: {valid_loss}")
