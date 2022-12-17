from pathlib import Path
import numpy as np
from mdgraph.models.gnn_lstm.inference import generate_embeddings
from mpro.analysis.projections import run_tsne
from mpro.plot.utils import plot_tsne


#output, loss = generate_embeddings(
#    model_cfg_path="/homes/abrace/src/pytorch-geometric-sandbox/test_plots/production-gnn-lstm-run-1/args.json",
#    h5_file="/homes/abrace/data/nsp10_16/nsp1016_448_cs1_3.h5",
#    model_weights_path="/homes/abrace/src/pytorch-geometric-sandbox/test_plots/production-gnn-lstm-run-1/checkpoints/epoch-50-20210302-191523.pt",
#    inference_batch_size=400,
#)

output, loss = generate_embeddings(
    model_cfg_path="/homes/abrace/src/pytorch-geometric-sandbox/test_plots/gnn-lstm--nsp1016-run-0/args.json",
    h5_file="/homes/abrace/src/pytorch-geometric-sandbox/test/data/1FME-1.h5", #"/homes/abrace/data/nsp10_16/nsp1016_448_cs1_3.h5",
    model_weights_path="/homes/abrace/src/pytorch-geometric-sandbox/test_plots/gnn-lstm--nsp1016-run-0/checkpoints/epoch-35-20210310-064651.pt",
    inference_batch_size=10213, #400,
)

#run_path = Path("/homes/abrace/src/pytorch-geometric-sandbox/nsp_inference_with_bba_model")
run_path = Path("/homes/abrace/src/pytorch-geometric-sandbox/test_plots/gnn-lstm--nsp1016-run-0/1FME-1_results")
run_path.mkdir()

saved = False
if not saved:
    np.save(run_path / "graph_embeddings.npy", output["graph_embeddings"])
    np.save(run_path / "node_embeddings.npy", output["node_embeddings"])
    np.save(run_path / "node_labels.npy", output["node_labels"])
    np.save(run_path / "rmsd.npy", output["rmsd"])
else:
    output = {
        "graph_embeddings": np.load(run_path / "graph_embeddings.npy"),
        "node_embeddings": np.load(run_path / "node_embeddings.npy"),
        "node_labels": np.load(run_path / "node_labels.npy"),
        "rmsd": np.load(run_path / "rmsd.npy"),
    }
