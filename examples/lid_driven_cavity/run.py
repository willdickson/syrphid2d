import pathlib
import syrphid2d
import jax.numpy as jnp
import numpy as np

filename = 'config.toml'
sim = syrphid2d.Simulation(filename)
sim.config.print()
sim.run()


















