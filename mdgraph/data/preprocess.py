import MDAnalysis
import numpy as np
from sklearn import preprocessing


def aminoacid_int_encoding(pdb_file):
    u = MDAnalysis.Universe(pdb_file)
    resnames = [r.resname for r in u.atoms.residues]
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(resnames)
    return resnames, labels


def aminoacid_int_to_onehot(labels):
    total_aa = np.max(labels) + 1 
    onehot = np.zeros((len(labels), total_aa))
    for i, label in enumerate(labels):
        onehot[i][label] = 1
    return onehot


def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full
