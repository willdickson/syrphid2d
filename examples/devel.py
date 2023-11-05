import pathlib
import syrphid2d
import jax.numpy as jnp
import numpy as np

#filename = 'lid_driven_cavity.toml'
filename = 'pipe_flow.toml'

sim = syrphid2d.Simulation(filename)
sim.config.print()
sim.run()


















