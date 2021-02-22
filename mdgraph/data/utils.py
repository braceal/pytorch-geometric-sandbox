import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict

PathLike = Union[Path, str]


def concatenate_h5(
    input_file_names: List[str], output_name: str, fields: Optional[List[str]] = None
):

    if not fields:
        # Peak into first file and collect all the field names
        with h5py.File(input_file_names[0], "r") as h5_file:
            fields = list(h5_file.keys())

    # Initialize data buffers
    data = {x: [] for x in fields}

    for in_file in input_file_names:
        with h5py.File(in_file, "r", libver="latest") as fin:
            for field in fields:
                data[field].append(fin[field][...])

    # Concatenate data
    for field in data:
        data[field] = np.concatenate(data[field])

    # Create new dsets from concatenated dataset
    fout = h5py.File(output_name, "w", libver="latest")
    for field, concat_dset in data.items():

        shape = concat_dset.shape
        chunkshape = (1,) + shape[1:]
        # Create dataset
        if concat_dset.dtype != np.object:
            if np.any(np.isnan(concat_dset)):
                raise ValueError("NaN detected in concat_dset.")
            dtype = concat_dset.dtype
        else:
            if field == "contact_map":  # contact_map is integer valued
                dtype = h5py.vlen_dtype(np.int16)
            else:
                dtype = h5py.vlen_dtype(np.float32)

        dset = fout.create_dataset(field, shape, chunks=chunkshape, dtype=dtype)
        # write data
        dset[...] = concat_dset[...]

    # Clean up
    fout.flush()
    fout.close()


def parse_h5(path: PathLike, fields: List[str]) -> Dict[str, np.ndarray]:
    r"""Helper function for accessing data fields in H5 file.
    Parameters
    ----------
    path : Union[Path, str]
        Path to HDF5 file.
    fields : List[str]
        List of dataset field names inside of the HDF5 file.
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary maping each field name in `fields` to a numpy
        array containing the data from the associated HDF5 dataset.
    """
    with h5py.File(path, "r") as f:
        return {field: f[field][...] for field in fields}
