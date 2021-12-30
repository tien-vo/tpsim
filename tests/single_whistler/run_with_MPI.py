r"""
    Main script for running simulation. Include processing functions here.
"""


from mpi4py import MPI
import matplotlib.pyplot as plt
import deck as d
import tpsim as tp
import numpy as np
import os


####################################################
# MPI parameters
####################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#######################################################
# Assignments to reduce namespace in loop
#######################################################
# Pusher
advance = tp.advance
# Model
EM_model = d.EM_model
# Time
Nt = d.Nt
dt = d.dt
save_interval = d.save_interval


####################################################
# Automatically ran function
####################################################
__message__ = f"""
------------------------------------------------------------------
    Simulation of particle dynamics in a single uniform whistler.
------------------------------------------------------------------
 * Job spawned on {size} processes.
----------
 * dtwce                                    : {dt:.4f}
 * Number of time steps                     : {Nt}
----------
 * Number of particles                      : {d.Np}
----------
 * Background magnetic field                : {d.B0} nT
 * Background electron density              : {d.n} cm-3
 * Electron plasma frequency                : {d.wpe:.4f} rad/s
 * Electron cyclotron frequency             : {d.wce:.4f} rad/s
 * wpe / wce                                : {d.wpe / d.wce:.4f}
----------
 * Original wave electric field amplitude   : {d._Ew0:.4f} mV/m
 * Original wave magnetic field amplitude   : {d._Bw0:.4f} nT
 * Wave frequency                           : {d.w_wce} wce
 * Wave obliquity                           : {np.degrees(d.theta)} deg
------------------------------------------------------------------
"""


def greet():
    """Print simulation information"""
    if rank == 0:
        print(__message__)


## Run functions
# Set up directories
if rank == 0:
    for directory in ["logs", "particles"]:
        if not os.path.exists(directory):
            os.mkdir(directory)


comm.Barrier()
# Logger
greet()
log = tp.logger(f"Rank {rank}", f"{rank}")
log.info(f"\n{__message__}")
# Print info greet()
comm.Barrier()
# Scatter particles' initial conditions
t = d.t_start
# Counts and disps are necessary to rebuild global arrays
counts, disps, x, y, z, ux, uy, uz = tp.scatter_ICs(
    d.xb, d.yb, d.zb, d.uxb, d.uyb, d.uzb, comm, MPI
)
log.info("Done scattering particles to MPI processes.")
if rank == 0:
    # Write to particles
    tp.dump_particles(0, d.t_start, d.xb, d.yb, d.zb, d.uxb, d.uyb, d.uzb)


if __name__ == "__main__":

    log.info(f"Starting main loop")
    for n in range(1, Nt + 1):

        ## Advance particles
        t, x, y, z, ux, uy, uz = advance(t, x, y, z, ux, uy, uz, EM_model, dt)

        ## Do other stuff here
        # At save interval
        if n % save_interval == 0:
            log.info(f"Pushed {n} steps")
            X, Y, Z, UX, UY, UZ = tp.gather_particles(
                x, y, z, ux, uy, uz, counts, disps, comm, MPI
            )
            if rank == 0:
                tp.dump_particles(n, t, X, Y, Z, UX, UY, UZ)

