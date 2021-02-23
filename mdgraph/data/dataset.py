import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset
from torch_geometric.data import Data
from mdgraph.data.preprocess import aminoacid_int_to_onehot


PathLike = Union[str, Path]


class ContactMapDataset(Dataset):
    """
    PyTorch Dataset class to load contact matrix data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """

    def __init__(
        self,
        path: PathLike,
        dataset_name: str,
        scalar_dset_names: List[str],
        node_feature: str = "amino_acid_onehot",
        constant_num_node_features: int = 20,
        scalar_requires_grad: bool = False,
    ):
        """
        Parameters
        ----------
        path : PathLike
            Path to h5 file containing contact matrices.
        dataset_name : str
            Path to contact maps in HDF5 file.
        scalar_dset_names : List[str]
            List of scalar dataset names inside HDF5 file to be passed
            to training logs.
        node_feature : str
            Type of node features to use. Available options are `constant`,
            `identity`, and `amino_acid_onehot`. If `constant` is selected,
            `constant_num_node_features` must be selected.
        constant_num_node_features : int
            Number of node features when using constant `node_feature` vectors.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            `scalar_dset_names`. Set to True, to use scalars for multi-task
            learning. If scalars are only required for plotting, then set it as False.
        """
        self._file_path = str(path)
        self._dataset_name = dataset_name
        self._scalar_dset_names = scalar_dset_names
        self._constant_num_node_features = constant_num_node_features
        self._scalar_requires_grad = scalar_requires_grad
        self._initialized = False

        # Get length and labels
        with self._open_h5_file() as f:
            self._labels = f["amino_acids"][...]
            self._len = len(f[self._dataset_name])

        self.num_nodes = len(self._labels)
        self._node_features = self._select_node_features(node_feature)

        # Convert to torch.Tensor
        self._node_features = torch.from_numpy(self._node_features).to(torch.float32)
        self._labels = torch.from_numpy(self._labels).to(torch.long)

    def _select_node_features(self, node_feature: str) -> np.ndarray:
        if node_feature == "constant":
            node_features = np.ones((self.num_nodes, self._constant_num_node_features))
        elif node_feature == "identity":
            node_features = np.eye(self.num_nodes)
        elif node_feature == "amino_acid_onehot":
            node_features = aminoacid_int_to_onehot(self._labels)
        else:
            raise ValueError(f"node_feature: {node_feature} not supported.")
        return node_features

    def _open_h5_file(self):
        return h5py.File(self._file_path, "r", libver="latest", swmr=False)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        # Only happens once. Need to open h5 file in current process
        if not self._initialized:
            self._h5_file = self._open_h5_file()
            self.dset = self._h5_file[self._dataset_name]
            # Load scalar dsets
            self.scalar_dsets = {
                name: self._h5_file[name] for name in self._scalar_dset_names
            }
            self._initialized = True

        # Get adjacency list
        edge_index = self.dset[idx, ...].reshape(2, -1)  # [2, num_edges]
        edge_index = torch.from_numpy(edge_index).to(torch.long)

        sample = {}

        # Graph data object
        data = Data(x=self._node_features, edge_index=edge_index, y=self._labels)
        data.num_nodes = self.num_nodes
        sample["data"] = data
        # Add index into dataset to sample
        sample["index"] = torch.tensor(idx, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self._scalar_requires_grad
            )

        return sample
