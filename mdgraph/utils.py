import time
import json
from pathlib import Path
from typing import Dict, Any
import torch
import pandas as pd
from plotly.io import to_html
from molecules.plot.tsne import compute_tsne, plot_tsne_plotly


def tsne_validation(embeddings, paint, paint_name, plot_dir, plot_name):
    print(f"t-SNE on input shape {embeddings.shape}")
    tsne_embeddings = compute_tsne(embeddings)
    fig = plot_tsne_plotly(
        tsne_embeddings, df_dict={paint_name: paint}, color=paint_name
    )
    html_string = to_html(fig)
    time_stamp = time.strftime(f"{plot_name}-%Y%m%d-%H%M%S.html")
    with open(plot_dir.joinpath(time_stamp), "w") as f:
        f.write(html_string)


def log_epoch_stats(epoch: int, stats: Dict[str, float], out_file: Path):
    df = pd.DataFrame({key: [val] for key, val in stats.items()})
    df["epoch"] = [epoch]
    df.set_index("epoch", inplace=True)
    if epoch == 1:
        df.to_csv(out_file)
    else:
        df.to_csv(out_file, mode="a", header=False)


def log_checkpoint(epoch, checkpoint: Dict[str, Any], checkpoint_dir: Path):
    time_stamp = time.strftime(f"epoch-{epoch}-%Y%m%d-%H%M%S.pt")
    path = checkpoint_dir.joinpath(time_stamp)
    torch.save(checkpoint, path)


def log_args(args_dict: dict, out_file: Path):
    """Save file containing argparse commands for documenting runs."""
    with open("commandline_args.txt", "w") as f:
        json.dump(args_dict, f, indent=2)
