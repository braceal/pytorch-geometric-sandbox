import argparse
import json
from typing import Union
from pathlib import Path

PathLike = Union[str, Path]


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variational_node", action="store_true", help="Use variational node encoder"
    )
    parser.add_argument(
        "--variational_lstm",
        action="store_true",
        help="Use variational LSTM Autoencoder",
    )
    parser.add_argument(
        "--graph_attention",
        action="store_true",
        help="Use GAT network for node encoder.",
    )
    parser.add_argument(
        "--node_recon_loss",
        action="store_true",
        help="Add node embedding reconstruction loss.",
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
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=1,
        help="Number of LSTM layers for encoder and decoder.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Whether LSTM should be bidirectional.",
    )

    parser.add_argument(
        "--tsne_interval",
        type=int,
        default=5,
        help="Run t-SNE every `tsne_interval` epochs.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default="./test_plots",
        help="Output directory for model results.",
    )
    parser.add_argument(
        "-f", "--jupyter", default="jupyter", help="For jupyter compatability"
    )

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args


def args_from_json(args_json: PathLike):
    parser = get_parser()  # argparse.ArgumentParser()
    with open(args_json, "r") as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    return args
