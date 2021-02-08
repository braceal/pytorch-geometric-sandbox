import MDAnalysis
from sklearn import preprocessing

def aminoacid_1hot(pdb_file):
    u = MDAnalysis.Universe(pdb_file)
    resnames = [r.resname for r in u.atoms.residues]
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(resnames)
    return resnames, labels
