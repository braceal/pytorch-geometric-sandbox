import MDAnalysis


def aminoacid_1hot(pdb_file):
    u = MDAnalysis.Universe(pdb_file)
    residues = u.atoms.residues
    print(residues)
    return residues
