import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from torch.utils.data import Dataset

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
        shape: Tuple[int, ...],
        split_ptc: float = 0.8,
        split: str = "train",
        seed: int = 333,
        scalar_requires_grad: bool = False,
        values_dset_name: Optional[str] = None,
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
        shape : tuple
            Shape of contact matrices (H, W), may be (1, H, W).
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
        values_dset_name: str, optional
            Name of HDF5 dataset field containing optional values of the entries
            the distance/contact matrix. By default, values are all assumed to be 1
            corresponding to a binary contact map and created on the fly.
        """
        if split not in ("train", "valid"):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError("Parameter split_ptc must satisfy 0 <= split_ptc <= 1.")

        # HDF5 data params
        self.file_path = str(path)
        self.dataset_name = dataset_name
        self.scalar_dset_names = scalar_dset_names
        self.shape = shape
        self._scalar_requires_grad = scalar_requires_grad
        self._values_dset_name = values_dset_name

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
            if self._values_dset_name is not None:
                self.val_dset = self._h5_file[self._values_dset_name]
            # Load scalar dsets
            self.scalar_dsets = {
                name: self._h5_file[name] for name in self.scalar_dset_names
            }
            self._initialized = True

        # get real index
        index = self.indices[idx]

        ind = self.dset[index, ...].reshape(2, -1)
        indices = torch.from_numpy(ind).to(torch.long)
        # Create array of 1s, all values in the contact map are 1. Or load values.
        if self._values_dset_name is not None:
            values = torch.from_numpy(self.val_dset[index, ...]).to(torch.float32)
        else:
            values = torch.ones(indices.shape[1], dtype=torch.float32)
        # Set shape to the last 2 elements of self.shape.
        # Handles (1, W, H) and (W, H)
        data = torch.sparse.FloatTensor(indices, values, self.shape[-2:]).to_dense()
        data = data.view(self.shape)

        sample = {"X": data}
        # Add index into dataset to sample
        sample["index"] = torch.tensor(index, requires_grad=False)
        # Add scalars
        for name, dset in self.scalar_dsets.items():
            sample[name] = torch.tensor(
                dset[index], requires_grad=self._scalar_requires_grad
            )

        return sample
