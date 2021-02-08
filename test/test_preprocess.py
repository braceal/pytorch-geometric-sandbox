from pathlib import Path
import numpy as np
from mdgraph.data.preprocess import aminoacid_int_encoding, aminoacid_int_to_onehot


TEST_DATA_PATH = Path(__file__).parent / "data/1FME-unfolded.pdb"


def test_residue_onehot_encoding():
    residues, labels = aminoacid_int_encoding(str(TEST_DATA_PATH))
    assert len(residues) == 28
    assert all(isinstance(r, str) for r in residues)
    num_unique_aa = len(np.unique(labels))
    onehot = aminoacid_int_to_onehot(labels)
    assert onehot.shape == (len(labels), num_unique_aa)
    # np.save("onehot_bba_amino_acid_labels.npy", labels)
