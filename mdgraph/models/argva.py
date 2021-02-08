from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from mdgraph.dataset import ContactMapDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ARGVA


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
num_features = 5
path = Path(__file__).parent / "../../test/data/BBA-subset-100.h5"
dataset = ContactMapDataset(path, "contact_map", ["rmsd"], num_features)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Models
encoder = Encoder(num_features, hidden_channels=32, out_channels=32)
discriminator = Discriminator(in_channels=32, hidden_channels=64, out_channels=32)
model = ARGVA(encoder, discriminator)

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizers
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)


def train():
    train_loss = 0.0
    for sample in loader:
        model.train()
        encoder_optimizer.zero_grad()
        data = sample["X"]
        data = data.to(device)
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
        train_loss += loss.item()
    train_loss /= len(loader)
    return train_loss


print(f"Traning on {len(loader)} examples")

for epoch in range(1, 151):
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.3f}")
