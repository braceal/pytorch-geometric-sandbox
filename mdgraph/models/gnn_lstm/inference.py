from pathlib import Path
from typing import Union, Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.data import DataLoader
from mdgraph.data.dataset import ContactMapDataset
from mdgraph.models.gnn_lstm.config import args_from_json
from mdgraph.models.gnn_lstm.model import (
    VariationalGCNEncoder,
    GATEncoder,
    GCNEncoder,
    LSTM_AE,
    validate_with_rmsd,
)

PathLike = Union[str, Path]


def generate_embeddings(
    model_cfg_path: PathLike,
    h5_file: PathLike,
    model_weights_path: PathLike,
    inference_batch_size: int,
) -> Tuple[Dict[str, np.ndarray], float]:

    # TODO: Make parameters configurable

    # Parameters
    out_channels = 10
    num_features = 20
    lstm_latent_dim = 10

    args = args_from_json(model_cfg_path)

    dataset = ContactMapDataset(h5_file, "contact_map", ["rmsd"])
    data_loader = DataLoader(dataset, inference_batch_size)
    num_nodes = dataset.num_nodes

    node_emb_recon_criterion = nn.MSELoss() if args.node_recon_loss else None

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

    checkpoint = torch.load(model_weights_path, map_location="cpu")
    node_encoder.load_state_dict(checkpoint["node_encoder_state_dict"])
    lstm_ae.load_state_dict(checkpoint["lstm_ae_state_dict"])

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_encoder = node_encoder.to(device)
    node_decoder = node_decoder.to(device)
    lstm_ae = lstm_ae.to(device)

    output, total_loss = validate_with_rmsd(
        node_encoder,
        node_decoder,
        lstm_ae,
        data_loader,
        device,
        inference_batch_size,
        num_nodes,
        out_channels,
        lstm_latent_dim,
        args.use_node_z,
        args.node_recon_loss,
        args.variational_node,
        args.variational_lstm,
        node_emb_recon_criterion,
    )

    return output, total_loss
