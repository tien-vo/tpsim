"""
    MPI helpers.
"""


__all__ = [
    "chunk",
    "scatter_ICs",
    "gather_particles",
    "scatter_arrays",
    "gather_arrays",
]


import numpy as np
import warnings


def chunk(data_size, world_size):
    r"""Calculate how to separate the data_size into chunks from the world_size.
    NOTE: This only works for 1D arrays (let's keep it simple for now).

    ----------
    Parameters
    ----------
    data_size: int
        The size of the large array
    world_size: int
        The size of the world (the maximum number of arrays we can split the
        data_size into)

    -------
    Returns
    -------
    counts: int, array_like
        Number of elements each core gets (the size of the smaller arrays)
    displacements: int, array_like
        The starting indices in the master data array
    """
    # Temporary array
    _tmp = np.arange(data_size, dtype=np.float64)
    split = np.array_split(_tmp, world_size, axis=0)
    counts = np.array([len(item) for item in split])
    displacements = np.insert(np.cumsum(counts), 0, 0)[0:-1]
    return counts, displacements


def scatter_ICs(X, Y, Z, UX, UY, UZ, comm, MPI):
    r"""Scatter the six state vector components to each MPI rank through
    chunking.

    ----------
    Parameters
    ----------
    X: (N,) array_like
        The X position of N particles
    Y: (N,) array_like
        The Y position of N particles
    Z: (N,) array_like
        The Z position of N particles
    UX: (N,) array_like
        The X velocity of N particles
    UY: (N,) array_like
        The Y velocity of N particles
    UZ: (N,) array_like
        The Z velocity of N particles
    comm: MPI.COMM_WORLD
        The world's communicator from mpi4py
    MPI: MPI
        The MPI module of mpi4py

    -------
    Returns
    -------
    counts: int, array_like
        Number of elements each core gets (the size of the smaller arrays)
    displacements: int, array_like
        The starting indices in the master data array
    x: array_like
        The local x position split into rank n. The size is counts[n]
    y: array_like
        The local y position split into rank n. The size is counts[n]
    z: array_like
        The local z position split into rank n. The size is counts[n]
    ux: array_like
        The local x velocity split into rank n. The size is counts[n]
    uy: array_like
        The local y velocity split into rank n. The size is counts[n]
    uz: array_like
        The local z velocity split into rank n. The size is counts[n]
    """
    # Get global size
    Np_global = X.size
    # Quick test to assure same array sizes
    assert X.size == Y.size == Z.size == UX.size == UY.size == UZ.size
    # Chunk data
    counts, disps = chunk(Np_global, comm.Get_size())
    # Get local size
    Np_local = counts[comm.Get_rank()]
    if Np_local > 1e4:
        warnings.warn("Might want to increase the number of processes")
    # Preallocate
    x, y, z, ux, uy, uz = np.zeros((6, Np_local), dtype=np.float64)
    # Scatter
    comm.Scatterv([X, counts, disps, MPI.DOUBLE], x)
    comm.Scatterv([Y, counts, disps, MPI.DOUBLE], y)
    comm.Scatterv([Z, counts, disps, MPI.DOUBLE], z)
    comm.Scatterv([UX, counts, disps, MPI.DOUBLE], ux)
    comm.Scatterv([UY, counts, disps, MPI.DOUBLE], uy)
    comm.Scatterv([UZ, counts, disps, MPI.DOUBLE], uz)
    return counts, disps, x, y, z, ux, uy, uz


def gather_particles(x, y, z, ux, uy, uz, counts, disps, comm, MPI):
    r"""Gather the local state vectors from each MPI rank and merge into
    a global array. This is the opposite operation from scatter_ICs

    ----------
    Parameters
    ----------
    x: (N,) array_like
        The X position of N particles
    y: (N,) array_like
        The Y position of N particles
    z: (N,) array_like
        The Z position of N particles
    ux: (N,) array_like
        The X velocity of N particles
    uy: (N,) array_like
        The Y velocity of N particles
    uz: (N,) array_like
        The Z velocity of N particles
    counts: int, array_like
        Number of elements each core gets (the size of the local arrays)
    displacements: int, array_like
        The starting indices in the master data array
    comm: MPI.COMM_WORLD
        The world's communicator from mpi4py
    MPI: MPI
        The MPI module of mpi4py

    -------
    Returns
    -------
    X: array_like
        The merged x position from all ranks
    Y: array_like
        The merged y position from all ranks
    Z: array_like
        The merged z position from all ranks
    UX: array_like
        The merged x velocity from all ranks
    UY: array_like
        The merged y velocity from all ranks
    UZ: array_like
        The merged z velocity from all ranks
    """
    # Wait for all processes to get to this point
    rank = comm.Get_rank()
    comm.Barrier()
    # Get global size
    Np_global = np.sum(counts)
    # Preallocate
    if rank == 0:
        X, Y, Z, UX, UY, UZ = np.zeros((6, Np_global), dtype=np.float64)
    else:
        X, Y, Z, UX, UY, UZ = None, None, None, None, None, None
    # Gather
    comm.Gatherv(x, [X, counts, disps, MPI.DOUBLE])
    comm.Gatherv(y, [Y, counts, disps, MPI.DOUBLE])
    comm.Gatherv(z, [Z, counts, disps, MPI.DOUBLE])
    comm.Gatherv(ux, [UX, counts, disps, MPI.DOUBLE])
    comm.Gatherv(uy, [UY, counts, disps, MPI.DOUBLE])
    comm.Gatherv(uz, [UZ, counts, disps, MPI.DOUBLE])
    return X, Y, Z, UX, UY, UZ


def scatter_arrays(arrays, comm, MPI):
    r"""Scatter the second dimension of the variable arrays to each MPI
    rank through chunking. This is a generalization of `scatter_ICs`.

    ----------
    Parameters
    ----------
    arrays: (Nc, N) array_like
        An array of `Nc` quantities relating to N particles. The chunking
        will be done in the second dimension.
    comm: MPI.COMM_WORLD
        The world's communicator from mpi4py
    MPI: MPI
        The MPI module of mpi4py

    -------
    Returns
    -------
    counts: int, array_like
        Number of elements each core gets (the size of the smaller arrays)
    displacements: int, array_like
        The starting indices in the master data array
    local_arrays: (Nc, n) array_like
        The Nc local arrays split into rank n. The size is counts[n]
    """
    # Get global size
    Np_global = arrays[0].shape[0]
    # Test to assure same global size
    for array in arrays[1:]:
        assert array.shape[0] == Np_global

    # Chunk data
    counts, disps = chunk(Np_global, comm.Get_size())
    # Get local size
    Np_local = counts[comm.Get_rank()]
    if Np_local > 1e4:
        warnings.warn("Might want to increase the number of processes")
    local_arrays = []
    for array in arrays:
        # Preallocate
        local_array = np.zeros(Np_local, dtype=np.float64)
        comm.Scatterv([array, counts, disps, MPI.DOUBLE], local_array)
        local_arrays.append(local_array)

    return (counts, disps, *local_arrays)


def gather_arrays(arrays, counts, disps, comm, MPI):
    r"""Gather the local state vectors from each MPI rank and merge into
    a global array. This is the opposite operation from `gather_particles`.

    ----------
    Parameters
    ----------
    arrays: (Nc, n) array_like
        A local array in rank n. The merging will be done in the second
        dimension.
    counts: int, array_like
        Number of elements each core gets (the size of the local arrays)
    displacements: int, array_like
        The starting indices in the master data array
    comm: MPI.COMM_WORLD
        The world's communicator from mpi4py
    MPI: MPI
        The MPI module of mpi4py

    -------
    Returns
    -------
    global_arrays: (Nc, N) array_like
        A merged array of `Nc` quantities relating to N particles from all
        ranks.
    """
    # Wait for all processes to get to this point
    rank = comm.Get_rank()
    comm.Barrier()
    # Get global size
    Np_global = np.sum(counts)
    global_arrays = []
    for array in arrays:
        # Preallocate
        if rank == 0:
            global_array = np.zeros(Np_global, dtype=np.float64)
        else:
            global_array = None
        # Gather
        comm.Gatherv(array, [global_array, counts, disps, MPI.DOUBLE])
        global_arrays.append(global_array)

    return global_arrays

