import pathlib
import syrphid2d
import jax.numpy as jnp
import numpy as np

sim = syrphid2d.Simulation('lid_driven_cavity.toml')
sim.config.print()
sim.run()


















