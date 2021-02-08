import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Union
from torch.utils.data import Dataset
from torch_geometric.data import Data


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
        num_node_features: int,
        split_ptc: float = 0.8,
        split: str = "train",
        seed: int = 333,
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
        split_ptc : float
            Percentage of total data to be used as training set.
        split : str
            Either 'train' or 'valid', specifies whether this
            dataset returns train or validation data.
        seed : int
            Seed for the RNG for the splitting. Make sure it is the
            same for all workers reading from the same file.
        scalar_requires_grad : bool
            Sets requires_grad torch.Tensor parameter for scalars specified by
            `scalar_dset_names`. Set to True, to use scalars for multi-task
            learning. If scalars are only required for plotting, then set it as False.
        """
        if split not in ("train", "valid"):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError("Parameter split_ptc must satisfy 0 <= split_ptc <= 1.")

        # HDF5 data params
        self.file_path = str(path)
        self.dataset_name = dataset_name
        self.scalar_dset_names = scalar_dset_names
        self._num_node_features = num_node_features
        self._scalar_requires_grad = scalar_requires_grad

        # get lengths and paths
        with self._open_h5_file() as f:
            self.len = len(f[self.dataset_name])

        # do splitting
        self.split_ind = int(split_ptc * self.len)
        self.split = split
        split_rng = np.random.default_rng(seed)
        self.indices = split_rng.permutation(list(range(self.len)))
        if self.split == "train":
            self.indices = sorted(self.indices[: self.split_ind])
        else:
            self.indices = sorted(self.indices[self.split_ind :])

        # inited:
        self._initialized = False

    def _open_h5_file(self):
        return h5py.File(self.file_path, "r", libver="latest", swmr=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # Only happens once. Need to open h5 file in current process
        if not self._initialized:
            self._h5_file = self._open_h5_file()
            self.dset = self._h5_file[self.dataset_name]
            # Load scalar dsets
            self.scalar_dsets = {
                name: self._h5_file[name] for name in self.scalar_dset_names
            }
            self._initialized = True

        # get real index
        index = self.indices[idx]

        # Get adjacency list
        edge_index = self.dset[index, ...].reshape(2, -1)  # [2, num_edges]
        edge_index = torch.from_numpy(edge_index).to(torch.long)

        # node features (contast)
        x = np.ones((edge_index.shape[1], self._num_node_features))

        # Great graph data object
        data = Data(x=x, edge_index=edge_index)

        sample = {"X": data}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(index, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[index], requires_grad=self._scalar_requires_grad
            )

        return sample
