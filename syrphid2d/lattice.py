from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class Lattice:

    def __init__(self):
        self.num = 9                 # Number of lattice vectors
        self.cs = 1/jnp.sqrt(3.0)    # Lattice speed of sound
        self.a0 = 1/(self.cs**2)     # 1st equilib function constant 
        self.a1 = 1/(2*self.cs**4)   # 2nd equilib function constant 
        self.a2 = 1/(2*self.cs**2)   # 3rd equilib function constant 
        self.w = create_w()          # Array of lattice weights
        self.e = create_e()          # Array of lattice vectors

    @property
    def avals(self):
        return self.a0, self.a1, self.a2


def create_w(): 
    w = ( 4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36)
    return w


def create_e(): 
    e = ((0, 0), (1, 0), (0, 1), (-1, 0), (0,-1), (1, 1), (-1, 1), (-1,-1), ( 1,-1))
    return e
    
