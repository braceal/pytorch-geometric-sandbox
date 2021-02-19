import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
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


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

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


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# Data
path = Path(__file__).parent / "../../test/data/BBA-subset-100.h5"
node_feature_path = (
    Path(__file__).parent / "../../test/data/onehot_bba_amino_acid_labels.npy"
)
dataset = ContactMapDataset(
    path, "contact_map", ["rmsd"], node_feature_path=node_feature_path
)
data = dataset[0]["X"]
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Parameters
out_channels = 10
num_features = 13

# Model
if not args.variational:
    if not args.linear:
        model = GAE(GCNEncoder(num_features, out_channels))
    else:
        model = GAE(LinearEncoder(num_features, out_channels))
else:
    if args.linear:
        model = VGAE(VariationalLinearEncoder(num_features, out_channels))
    else:
        model = VGAE(VariationalGCNEncoder(num_features, out_channels))

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, data = model.to(device), data.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()

    if args.constant:
        num_nodes = int(data.edge_index.max().item()) + 1
        x = torch.ones((num_nodes, num_features))
        x = x.to(device)
    else:
        x = data.x

    z = model.encode(data.x, data.edge_index)
    loss = model.recon_loss(z, data.edge_index)
    if args.variational:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def validate_with_rmsd():
    model.eval()
    output = defaultdict(list)
    with torch.no_grad():
        for sample in tqdm(loader):
            data = sample["X"]
            data = data.to(device)

            if args.constant:
                num_nodes = int(data.edge_index.max().item()) + 1
                x = torch.ones((num_nodes, num_features))
                x = x.to(device)
            else:
                x = data.x

            z = model.encode(x, data.edge_index)

            # Collect embeddings for plot
            emb = z.detach().cpu().numpy()
            output["graph_embeddings"].append(emb.sum(axis=0))
            output["node_embeddings"].append(emb)
            output["node_labels"].append(data.y.detach().cpu().numpy())
            output["rmsd"].append(sample["rmsd"].detach().cpu().numpy())

    output = {key: np.array(val).squeeze() for key, val in output.items()}

    shape = output["node_embeddings"].shape
    output["node_embeddings"] = output["node_embeddings"].reshape(
        shape[0] * shape[1], -1
    )
    output["node_labels"] = output["node_labels"].flatten()

    return output


for epoch in range(1, args.epochs + 1):
    loss = train()
    print(f"Epoch: {epoch:03d}\tLoss: {loss}")

# Validate
output = validate_with_rmsd()
random_sample = np.random.choice(len(output["node_embeddings"]), 8000, replace=False)
tsne_validation(
    embeddings=output["node_embeddings"][random_sample],
    paint=output["node_labels"][random_sample],
    paint_name="node_labels",
    plot_dir=Path("./test_plots"),
    plot_name=f"epoch-{epoch}-node_embeddings",
)
