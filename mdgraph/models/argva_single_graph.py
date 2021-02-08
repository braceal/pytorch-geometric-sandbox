import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mdgraph.data.dataset import ContactMapDataset
from torch_geometric.nn import GCNConv, ARGVA
from torch_geometric.data import DataLoader
from molecules.plot.tsne import compute_tsne, plot_tsne_plotly
from plotly.io import to_html


def tsne_validation(embeddings, paint, paint_name, epoch, plot_dir):
    print(f"t-SNE on input shape {embeddings.shape}")
    tsne_embeddings = compute_tsne(embeddings)
    fig = plot_tsne_plotly(
        tsne_embeddings, df_dict={paint_name: paint}, color=paint_name
    )
    html_string = to_html(fig)
    time_stamp = time.strftime(
        f"t-SNE-plotly-{paint_name}-epoch-{epoch}-%Y%m%d-%H%M%S.html"
    )
    with open(plot_dir.joinpath(time_stamp), "w") as f:
        f.write(html_string)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


# Data
path = Path(__file__).parent / "../../test/data/BBA-subset-100.h5"
node_feature_path = Path(__file__).parent / "../../test/data/onehot_bba_amino_acid_labels.npy"
dataset = ContactMapDataset(
    path, "contact_map", ["rmsd"], node_feature_path=node_feature_path
)
data = dataset[0]["X"]
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Models
encoder = Encoder(data.num_features, hidden_channels=32, out_channels=10)
discriminator = Discriminator(in_channels=10, hidden_channels=64, out_channels=10)
model = ARGVA(encoder, discriminator)

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, data = model.to(device), data.to(device)

# Optimizers
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)


def train():
    model.train()
    encoder_optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)

    for _ in range(5):
        discriminator.train()
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z)
        discriminator_loss.backward()
        discriminator_optimizer.step()

    loss = model.recon_loss(z, data.edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    encoder_optimizer.step()
    return loss


def validate():
    model.eval()
    graph_embeddings = {"embeddings": [], "rmsd": []}
    with torch.no_grad():
        for sample in tqdm(loader):
            data = sample["X"]
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)

            # Collect embeddings for plot
            emb = z.detach().cpu().numpy().sum(axis=0)
            graph_embeddings["embeddings"].append(emb)
            graph_embeddings["rmsd"].append(sample["rmsd"].detach().cpu().numpy())

    graph_embeddings = {
        key: np.array(val).squeeze() for key, val in graph_embeddings.items()
    }

    return graph_embeddings


for epoch in range(1, 151):
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.3f}")

# Validate
graph_embeddings = validate()
tsne_validation(
    graph_embeddings["embeddings"],
    paint=graph_embeddings["rmsd"],
    paint_name="rmsd",
    epoch=epoch,
    plot_dir=Path("./plot"),
)
