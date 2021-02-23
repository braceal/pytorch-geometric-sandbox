import time
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import DataLoader
from mdgraph.data.dataset import ContactMapDataset
from mdgraph.utils import tsne_validation

parser = argparse.ArgumentParser()
parser.add_argument("--variational", action="store_true")
parser.add_argument("--linear", action="store_true")
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--name", type=str)
parser.add_argument("--constant", action="store_true")
args = parser.parse_args()

print("variational:", args.variational)
print("linear:", args.linear)
print("constant:", args.constant)

# TODO: try using the lstm-decoded node embeddings with the inner product
# decoder


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
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


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
num_features = 13
lstm_latent_dim = 10
batch_size = 128
num_nodes = 28

# Data
path = Path(__file__).parent / "../../test/data/BBA-subset-100.h5"
dataset = ContactMapDataset(path, "contact_map", ["rmsd"])
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Models
if not args.variational:
    if not args.linear:
        node_ae = GAE(GCNEncoder(num_features, out_channels))
    else:
        node_ae = GAE(LinearEncoder(num_features, out_channels))
else:
    if args.linear:
        node_ae = VGAE(VariationalLinearEncoder(num_features, out_channels))
    else:
        node_ae = VGAE(VariationalGCNEncoder(num_features, out_channels))

lstm_ae = LSTM_AE(out_channels, lstm_latent_dim)

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
node_ae, lstm_ae = node_ae.to(device), lstm_ae.to(device)

# Optimizer
optimizer = torch.optim.Adam(
    set(node_ae.parameters()) | set(lstm_ae.parameters()), lr=0.01
)

# Criterion
node_emb_recon_criterion = nn.MSELoss()


def train():
    node_ae.train()
    lstm_ae.train()

    train_loss = 0.0
    for i, sample in enumerate(loader):
        start = time.time()
        optimizer.zero_grad()

        data = sample["X"]
        data = data.to(device)

        node_z = node_ae.encode(data.x, data.edge_index)
        # print("node_z.shape:", node_z.shape)
        loss = node_ae.recon_loss(node_z, data.edge_index)
        if args.variational:
            loss = loss + (1 / data.num_nodes) * node_ae.kl_loss()

        # node_z = node_z.view(batch_size, -1, out_channels)
        node_z = node_z.view(-1, num_nodes, out_channels)
        graph_emb, node_z_recon = lstm_ae(node_z)
        # print(graph_emb.shape)
        # print(node_z_recon.shape)

        loss += node_emb_recon_criterion(node_z, node_z_recon)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if i % 1 == 0:
            print(
                f"Training {i}/{len(loader)}. Loss: {train_loss / (i + 1)} Batch time: {time.time() - start}"
            )

    train_loss /= len(loader)

    return train_loss


def validate_with_rmsd():
    node_ae.eval()
    lstm_ae.eval()
    output = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(loader):
            data = sample["X"]
            data = data.to(device)

            node_z = node_ae.encode(data.x, data.edge_index)
            node_z = node_z.view(-1, num_nodes, out_channels)
            graph_z, node_z_recon = lstm_ae(node_z)

            # Collect embeddings for plot
            node_emb = (
                node_z.view(batch_size * num_nodes, out_channels).detach().cpu().numpy()
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
        batch_size * len(loader), lstm_latent_dim
    )
    output["node_labels"] = output["node_labels"].flatten()

    for key, val in output.items():
        print(key, val.shape)

    return output


def validate(epoch: int):

    output = validate_with_rmsd()

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


for epoch in range(1, args.epochs + 1):
    loss = train()
    print(f"Epoch: {epoch:03d}\tLoss: {loss}")
    validate(epoch)
