from typing import Callable, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

EPS = 1e-15
MAX_LOGSTD = 10


def kld_loss(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    r"""Computes the KLD loss, either for the passed arguments :obj:`mu`
    and :obj:`logstd`.
    Args:
        mu (Tensor): The latent space for :math:`\mu`.
        logstd (Tensor): The latent space for :math:`\log\sigma`.
    """
    logstd = logstd.clamp(max=MAX_LOGSTD)
    return -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1)
    )


def reparametrize(
    mu: torch.Tensor, logstd: torch.Tensor, training: bool = False
) -> torch.Tensor:
    if training:
        return mu + torch.randn_like(logstd) * torch.exp(logstd)
    else:
        return mu


def recon_loss(decoder, z, pos_edge_index, neg_edge_index=None) -> torch.Tensor:
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


# TODO: refactor these GNN encoders into the same class


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GATEncoder(nn.Module):
    def __init__(self, num_features: int, out_channels: int):
        super(GATEncoder, self).__init__()

        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index) -> torch.Tensor:
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        # Cache these for computing kld_loss
        self.__mu__ = self.conv_mu(x, edge_index)
        self.__logstd__ = self.conv_logstd(x, edge_index).clamp(max=MAX_LOGSTD)
        z = reparametrize(self.__mu__, self.__logstd__, self.training)
        return z  # Might need to return __mu__ for reconstruction loss for the lstm decoder
        # TODO: Should mu, or the reparametrize vector be passed on to the LSTM?

    def kld_loss(self) -> torch.Tensor:
        return kld_loss(self.__mu__, self.__logstd__)


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

        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape BxNxD for B batches of N nodes
            by D node latent dimensions.
        """
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

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor, h_n: torch.Tensor, c_n: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape BxNxD for B batches of N nodes
            by D node latent dimensions.
        """
        batch_size, seq_len, input_size = x.shape

        outputs = torch.zeros_like(x)
        x_i = emb.view(batch_size, 1, input_size)

        # print("decoder x.shape:", x.shape)
        # print("decoder outputs.shape:", outputs.shape)
        for i in range(seq_len):
            # print("decoder x_i.shape:", x_i.shape)
            # print("decoder h_n.shape:", h_n.shape)
            # print("decoder c_n.shape:", c_n.shape)
            # print("decoder x_i.shape:", x_i.shape)
            output, (h_n, c_n) = self.lstm(x_i, (h_n, c_n))
            x_i = x[:, i].view(batch_size, 1, input_size)  # Single vector has seq_len=1
            # print("decoder x_i.shape:", x_i.shape)
            # print("decoder output.shape:", output.shape) # [512, 1, 10] # 10 is ^
            # print("decoder output.sqeeuze().shape", output.squeeze().shape) # [512, 10]

            # [:, : self.hidden_size] handles bidirectional and num_layers
            outputs[:, i, :] = output.squeeze()[:, : self.hidden_size]

        return outputs


class LSTM_AE(nn.Module):
    r"""LSTM Autoencoder model from: https://arxiv.org/pdf/1502.04681.pdf"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        variational: bool = False,
        **kwargs,
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
        variational : bool
            If True, becomes a variational encoder.
        """
        super().__init__()

        self.num_layers = num_layers
        self.variational = variational

        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, **kwargs)
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers, **kwargs)

        if self.variational:
            self.mu = nn.Linear(hidden_size, hidden_size)
            self.logstd = nn.Linear(hidden_size, hidden_size)

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_n, c_n = self.encoder(x)
        # Handle bidirectional and num_layers
        emb = h_n[self.num_layers - 1, ...]

        if self.variational:
            # Cache these for computing kld_loss
            self.__mu__, self.__logstd__ = self.mu(emb), self.logstd(emb)
            self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
            emb = reparametrize(self.__mu__, self.__logstd__, self.training)

        return emb, h_n, c_n

    def decode(
        self, x: torch.Tensor, emb: torch.Tensor, h_n: torch.Tensor, c_n: torch.Tensor
    ) -> torch.Tensor:
        # TODO: verify flip logic
        flipped_x = torch.flip(x, dims=(2,))
        decoded = self.decoder(flipped_x, emb, h_n, c_n)
        decoded = torch.flip(decoded, dims=(2,))
        return decoded

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # print("lstm ae: x.shape:", x.shape)
        emb, h_n, c_n = self.encode(x)
        decoded = self.decode(x, emb, h_n, c_n)
        return self.__mu__ if self.variational else emb, decoded

    def kld_loss(self) -> torch.Tensor:
        return kld_loss(self.__mu__, self.__logstd__)


def validate_with_rmsd(
    node_encoder: nn.Module,
    node_decoder: nn.Module,
    lstm_ae: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    batch_size: int,
    num_nodes: int,
    out_channels: int,
    lstm_latent_dim: int,
    use_node_z: bool,
    node_recon_loss: bool,
    variational_node: bool,
    variational_lstm: bool,
    node_emb_recon_criterion: Optional[Callable] = None,
) -> Tuple[Dict[str, np.ndarray], float]:
    node_encoder.eval()
    node_decoder.eval()
    lstm_ae.eval()
    output = defaultdict(list)
    total_loss = 0.0
    # return None, total_loss
    with torch.no_grad():
        for sample in data_loader:
            data = sample["data"]
            data = data.to(device)

            node_z = node_encoder(data.x, data.edge_index)

            graph_z, node_z_recon = lstm_ae(
                node_z.view(batch_size, num_nodes, out_channels)
            )
            node_z_recon = node_z_recon.view(batch_size * num_nodes, out_channels)

            # Reconstruction losses
            loss = recon_loss(
                node_decoder,
                node_z if use_node_z else node_z_recon,
                data.edge_index,
            )
            if node_recon_loss:
                assert node_emb_recon_criterion is not None
                loss += node_emb_recon_criterion(node_z, node_z_recon)

            # Variational losses
            if variational_node:
                loss += (1 / num_nodes) * node_encoder.kld_loss()
            if variational_lstm:
                loss += lstm_ae.kld_loss()

            total_loss += loss.item()

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
        batch_size * len(data_loader), lstm_latent_dim
    )
    output["node_labels"] = output["node_labels"].flatten()

    total_loss /= len(data_loader)
    return output, total_loss
