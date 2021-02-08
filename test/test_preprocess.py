from pathlib import Path
from mdgraph.data.preprocess import aminoacid_1hot


TEST_DATA_PATH = Path(__file__).parent / "data/1FME-unfolded.pdb"


def test_residue_list():
    residues = aminoacid_1hot(TEST_DATA_PATH)
    print(residues)
