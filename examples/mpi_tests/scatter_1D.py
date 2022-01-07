from mpi4py import MPI
from tpsim.mpi_helpers import chunk
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if __name__ == "__main__":

    dsize = 10
    if rank == 0:
        data = np.arange(dsize, dtype=np.float64)
        print(f"Data to scatter: {data}")
    else:
        data = None
        N = None

    counts, disps = chunk(dsize, size)
    local_data = np.zeros(counts[rank])

    comm.Scatterv([data, counts, disps, MPI.DOUBLE], local_data)
    comm.Barrier()

    print(f"Rank {rank} has {local_data}")

    processed_data = local_data ** 2
    gathered_data = np.zeros(dsize, dtype=np.float64) if rank == 0 else None
    comm.Gatherv(processed_data, [gathered_data, counts, disps, MPI.DOUBLE])
    if rank == 0:
        print(f"Gathered data is {gathered_data}")

