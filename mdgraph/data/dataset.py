import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
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
        num_node_features: Optional[int] = None,
        node_feature_path: Optional[PathLike] = None,
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
        num_node_features : int
            Number of node features.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            `scalar_dset_names`. Set to True, to use scalars for multi-task
            learning. If scalars are only required for plotting, then set it as False.
        """

        # HDF5 data params
        self._file_path = str(path)
        self._dataset_name = dataset_name
        self._scalar_dset_names = scalar_dset_names
        self._num_node_features = num_node_features
        self._scalar_requires_grad = scalar_requires_grad

        # get node features
        if node_feature_path is not None:
            self.labels = np.load(node_feature_path)
            self.node_features = aminoacid_int_to_onehot(self.labels)
            self.labels = torch.from_numpy(self.labels).to(torch.long)
            self.node_features = torch.from_numpy(self.node_features).to(torch.float32)
        else:
            self.node_features, self.labels = None, None

        # get lengths and paths
        with self._open_h5_file() as f:
            self._len = len(f[self._dataset_name])

        # inited:
        self._initialized = False

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

        # node features (contast all ones)
        if self.node_features is None:
            num_nodes = int(edge_index.max().item()) + 1
            x = torch.ones((num_nodes, self._num_node_features))
            y = None
        else:
            x = self.node_features
            y = self.labels

        # Great graph data object
        data = Data(x=x, edge_index=edge_index, y=y)

        sample = {"X": data}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(idx, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[idx], requires_grad=self._scalar_requires_grad
            )

        return sample
