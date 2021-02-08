from pathlib import Path
import numpy as np
from mdgraph.data.dataset import ContactMapDataset


TEST_DATA_PATH = Path(__file__).parent / "data/BBA-subset-100.h5"


def test_shapes():
    dataset = ContactMapDataset(TEST_DATA_PATH, "contact_map", ["rmsd"], 5)
    sample = dataset[0]
    assert sample["X"].num_features == 5
    assert sample["X"].num_nodes == 28


def test_dataloader():
    from torch_geometric.data import DataLoader

    dataset = ContactMapDataset(TEST_DATA_PATH, "contact_map", ["rmsd"], 5)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    assert len(loader) == np.ceil(len(dataset) / 32)


def test_amino_acid_features():
    amino_acid_features = (
        Path(__file__).parent / "data/onehot_bba_amino_acid_labels.npy"
    )
    dataset = ContactMapDataset(
        TEST_DATA_PATH, "contact_map", ["rmsd"], node_feature_path=amino_acid_features
    )
    sample = dataset[0]
    assert sample["X"].num_features == 5
    assert sample["X"].num_nodes == 28
    print(sample["X"].y)
