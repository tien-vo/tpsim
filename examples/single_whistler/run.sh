#!/usr/bin/bash

# Run simulation
mpirun python run.py

# Plot
python plot.py

# Make video
ffmpeg -y -r 40 -i plot_frames/%d.png -vb 20M velocity_space.mp4

