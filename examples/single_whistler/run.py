from mpi4py import MPI
import matplotlib.pyplot as plt
import tpsim as tp
import numpy as np
import deck as d
import os


# --------------------------------------------------------------------------- #
#                                MPI parameters
# --------------------------------------------------------------------------- #
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# --------------------------------------------------------------------------- #
#                                Pre-processing
# --------------------------------------------------------------------------- #
# Greet & set up directories for data
if rank == 0:
    print(d.sim_info)
    for directory in ["logs", "particles"]:
        if not os.path.exists(directory): os.mkdir(directory)

# Assignments to reduce namespace in loop
advance = tp.advance
EM_model = d.EM_model
Nt = d.Nt
dt = d.dt
save_interval = d.save_interval
# Logger
log = tp.logger(f"Rank {rank}", f"{rank}")
# ---------- Scatter initial conditions
comm.Barrier()
log.info("Job spawned on {size} processes.")
t = d.t_start
# Counts and disps are necessary for rebuilding global arrays
counts, disps, x, y, z, ux, uy, uz = tp.scatter_ICs(
    d.xn, d.yn, d.zn, d.uxn, d.uyn, d.uzn, comm, MPI
)
log.info("Done scattering particles to MPI processes.")
if rank == 0: tp.dump_particles(0, t, d.xn, d.yn, d.zn, d.uxn, d.uyn, d.uzn)


# --------------------------------------------------------------------------- #
#                                Run simulation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    log.info("Starting main loop")
    for n in range(1, Nt + 1):

        # Advance particles
        t, x, y, z, ux, uy, uz = advance(
            t, x, y, z, ux, uy, uz, EM_model, dt, s="e-"
        )

        # At save intervals ...
        if n % save_interval == 0:
            log.info(f"Pushed {n} steps")
            X, Y, Z, UX, UY, UZ = tp.gather_particles(
                x, y, z, ux, uy, uz, counts, disps, comm, MPI
            )
            # Save data
            if rank == 0: tp.dump_particles(n, t, X, Y, Z, UX, UY, UZ)

