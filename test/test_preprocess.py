from pathlib import Path
from mdgraph.data.preprocess import aminoacid_1hot


TEST_DATA_PATH = Path(__file__).parent / "data/1FME-unfolded.pdb"


def test_residue_list():
    residues, labels = aminoacid_1hot(str(TEST_DATA_PATH))
    assert len(residues) == 28
    assert all(isinstance(r, str) for r in residues)
