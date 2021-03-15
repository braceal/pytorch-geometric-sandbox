import time
from itertools import chain
import numpy as np
from collections import defaultdict
from typing import Tuple, Dict
import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.data import DataLoader
from mdgraph.data.dataset import ContactMapDataset
from mdgraph.models.gnn_lstm.config import get_args
from mdgraph.models.gnn_lstm.model import (
    VariationalGCNEncoder,
    GATEncoder,
    GCNEncoder,
    LSTM_AE,
    recon_loss,
)
from mdgraph.utils import tsne_validation, log_epoch_stats, log_checkpoint, log_args

# Parameters
out_channels = 10
num_features = 20
lstm_latent_dim = 10

# Setup run
args = get_args()
args.run_dir.mkdir()
args.run_dir.joinpath("checkpoints").mkdir()
args.run_dir.joinpath("plots").mkdir()
log_args(args.__dict__, args.run_dir.joinpath("args.json"))

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
elif args.graph_attention:
    node_encoder = GATEncoder(num_features, out_channels)
else:
    node_encoder = GCNEncoder(num_features, out_channels)
node_decoder = InnerProductDecoder()
lstm_ae = LSTM_AE(
    out_channels,
    lstm_latent_dim,
    num_layers=args.lstm_num_layers,
    bidirectional=args.bidirectional,
    variational=args.variational_lstm,
)

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


def train(epoch: int) -> float:
    node_encoder.train()
    node_decoder.train()
    lstm_ae.train()

    total_loss = 0.0
    for i, sample in enumerate(train_loader):
        # if i == 5:
        #    break
        start = time.time()
        optimizer.zero_grad()
        data = sample["data"]
        data = data.to(device)

        node_z = node_encoder(data.x, data.edge_index)
        # print("node_z.shape:", node_z.shape)
        graph_z, node_z_recon = lstm_ae(
            node_z.view(args.batch_size, num_nodes, out_channels)
        )
        node_z_recon = node_z_recon.view(args.batch_size * num_nodes, out_channels)
        # Reconstruction losses
        loss = recon_loss(
            node_decoder,
            node_z if args.use_node_z else node_z_recon,
            data.edge_index,
        )
        if args.node_recon_loss:
            loss += node_emb_recon_criterion(node_z, node_z_recon)

        # Variational losses
        if args.variational_node:
            loss += (1 / num_nodes) * node_encoder.kld_loss()
        if args.variational_lstm:
            loss += lstm_ae.kld_loss()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 100 == 0:
            print(
                f"Training Epoch: {epoch} Batch: {i}/{len(train_loader)} "
                f"Loss: {total_loss / (i + 1)} Batch time: {time.time() - start}"
            )

    total_loss /= len(train_loader)

    return total_loss


def validate_with_rmsd() -> Tuple[Dict[str, np.ndarray], float]:
    node_encoder.eval()
    node_decoder.eval()
    lstm_ae.eval()
    output = defaultdict(list)
    total_loss = 0.0
    # return None, total_loss
    with torch.no_grad():
        for sample in valid_loader:
            data = sample["data"]
            data = data.to(device)

            node_z = node_encoder(data.x, data.edge_index)

            graph_z, node_z_recon = lstm_ae(
                node_z.view(args.batch_size, num_nodes, out_channels)
            )
            node_z_recon = node_z_recon.view(args.batch_size * num_nodes, out_channels)

            # Reconstruction losses
            loss = recon_loss(
                node_decoder,
                node_z if args.use_node_z else node_z_recon,
                data.edge_index,
            )
            if args.node_recon_loss:
                loss += node_emb_recon_criterion(node_z, node_z_recon)

            # Variational losses
            if args.variational_node:
                loss += (1 / num_nodes) * node_encoder.kld_loss()
            if args.variational_lstm:
                loss += lstm_ae.kld_loss()

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


def validate(epoch: int) -> float:

    output, total_loss = validate_with_rmsd()

    if epoch % args.tsne_interval != 0:
        return total_loss

    # Paint graph embeddings
    random_sample = np.random.choice(
        len(output["graph_embeddings"]), 8000, replace=False
    )
    tsne_validation(
        embeddings=output["graph_embeddings"][random_sample],
        paint=output["rmsd"][random_sample],
        paint_name="rmsd",
        plot_dir=args.run_dir.joinpath("plots"),
        plot_name=f"epoch-{epoch}-graph_embeddings",
    )

    random_sample = np.random.choice(
        len(output["node_embeddings"]), 8000, replace=False
    )
    tsne_validation(
        embeddings=output["node_embeddings"][random_sample],
        paint=output["node_labels"][random_sample],
        paint_name="node_labels",
        plot_dir=args.run_dir.joinpath("plots"),
        plot_name=f"epoch-{epoch}-node_embeddings",
    )

    return total_loss


for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    valid_loss = validate(epoch)
    log_epoch_stats(
        epoch,
        {"train_loss": train_loss, "valid_loss": valid_loss},
        args.run_dir.joinpath("loss.csv"),
    )
    log_checkpoint(
        epoch,
        {
            "node_encoder_state_dict": node_encoder.state_dict(),
            "lstm_ae_state_dict": lstm_ae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        args.run_dir.joinpath("checkpoints"),
    )
    print(f"Epoch: {epoch:03d}\t Train Loss: {train_loss} Valid Loss: {valid_loss}")
