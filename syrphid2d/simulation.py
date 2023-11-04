import json
import pickle
import pathlib
import jax.numpy as jnp
from jax import jit
from .config import Config
from .lattice import Lattice
from .boundary import Boundary
from .state import initialize_state

class Simulation:

    def __init__(self, config_input):

        self.config = self.load_config(config_input)
        self.lattice = Lattice()
        self.boundary = Boundary(self.config)
        self.save_dir = None
        self.save_cnt = 0

        # Jit compiled versions of methods
        self.jit_update_iter = jit(self.update_iter)
        self.jit_set_boundary = jit(self.boundary.set)
        self.jit_set_boundaries = jit(self.set_boundaries)
        self.jit_predictor = jit(self.predictor)
        self.jit_corrector = jit(self.corrector)
        self.jit_ibcorrector = jit(self.ibcorrector)
        self.jit_isdone = jit(self.isdone)


    def run(self):
        self.setup_save()
        state = initialize_state(self.config, self.lattice)
        state = self.jit_set_boundaries(state)
        while not state['done']:
            state = self.jit_update_iter(state)
            state = self.jit_predictor(state)
            state = self.jit_corrector(state)
            state = self.jit_ibcorrector(state)
            state = self.jit_isdone(state)
            self.display(state)
            self.save(state)


    def set_boundaries(self,state):
        for name in state['vel']:
            state['vel'][name] = self.jit_set_boundary(state['vel'][name])
        return state


    def update_iter(self,state):
        state['n'] += 1
        state['t'] += self.config['mesh']['ds']
        return state


    def predictor(self, state):

        num_x, num_y = state['rho']['last'].shape
        avals = self.lattice.avals

        # Set last state to current state 
        state['rho']['last']      = state['rho']['curr']
        state['vel']['last']['u'] = state['vel']['curr']['u']
        state['vel']['last']['v'] = state['vel']['curr']['v']

        # Initialize matrices for new predictor state
        state['rho']['pred']      = jnp.zeros_like(state['rho']['pred'])
        state['vel']['pred']['u'] = jnp.zeros_like(state['vel']['pred']['u'])
        state['vel']['pred']['v'] = jnp.zeros_like(state['vel']['pred']['v'])

        # Get interior slices for rho, u and v predictor states
        rho_pred = state['rho']['pred'][1:-1, 1:-1]
        u_pred   = state['vel']['pred']['u'][1:-1, 1:-1]
        v_pred   = state['vel']['pred']['v'][1:-1, 1:-1]

        for e, w in zip(self.lattice.e, self.lattice.w):

            # Slice indices for equilibrium calculation
            ex, ey = e           # lattice basis vectors
            nx = 1 - ex          # slice start x index
            ny = 1 - ey          # slice start y index
            mx = num_x + nx - 2  # slice end x index
            my = num_y + ny - 2  # slice end y index

            # Get slices of current state for equilibrium calculation
            rho_curr = state['rho']['curr'][nx:mx, ny:my] 
            u_curr   = state['vel']['curr']['u'][nx:mx, ny:my]
            v_curr   = state['vel']['curr']['v'][nx:mx, ny:my]

            # Compute equilibrium function and update predictor rho, u and v  
            feq = equilib(rho_curr, u_curr, v_curr, e, w, *avals) 
            rho_pred += feq
            u_pred   += feq*ex
            v_pred   += feq*ey

        # Update state 
        state['rho']['pred']      = state['rho']['pred'].at[1:-1,1:-1].set(rho_pred)
        state['vel']['pred']['u'] = state['vel']['pred']['u'].at[1:-1,1:-1].set(u_pred)
        state['vel']['pred']['v'] = state['vel']['pred']['v'].at[1:-1,1:-1].set(v_pred)

        # Set boundary condition 
        state['vel']['pred']      = self.jit_set_boundary(state['vel']['curr'])

        return state

    def corrector(self, state):

        num_x, num_y = state['rho']['last'].shape
        avals = self.lattice.avals
        tau = state['tau']

        # Get interior slices for predictor velocity 
        rho_pred = state['rho']['pred'][1:-1, 1:-1]
        u_pred   = state['vel']['pred']['u'][1:-1, 1:-1]
        v_pred   = state['vel']['pred']['v'][1:-1, 1:-1]

        # Get combined density and velocity for corrector step
        rho_u_curr = rho_pred*u_pred
        rho_v_curr = rho_pred*v_pred

        for e, w in zip(self.lattice.e, self.lattice.w):

            # Slice indices
            ex, ey = e           # lattice basis vectors
            nx = 1 + ex          # slice start x index
            ny = 1 + ey          # slice start y index
            mx = num_x + nx - 2  # slice end x index
            my = num_y + ny - 2  # slice end y index

            # Get slices for rho, u and v for equilibrium calculation
            rho_pred = state['rho']['pred'][nx:mx, ny:my] 
            u_pred   = state['vel']['pred']['u'][nx:mx, ny:my] 
            v_pred   = state['vel']['pred']['v'][nx:mx, ny:my] 

            # Compute equilibrium function and update rho*u and rho*v
            feq = equilib(rho_pred, u_pred, v_pred, e, w, *avals) 
            rho_u_curr += (tau - 1)*feq*ex
            rho_v_curr += (tau - 1)*feq*ey

        # Get interior slices for rho, u and v last
        rho_last = state['rho']['last'][1:-1, 1:-1]
        u_last   = state['vel']['last']['u'][1:-1, 1:-1]
        v_last   = state['vel']['last']['v'][1:-1, 1:-1]

        # Update rho*u and rho*v current and divide by rho to get u and v
        rho_u_curr -= (tau - 1)*rho_last*u_last
        rho_v_curr -= (tau - 1)*rho_last*v_last
        u_curr = rho_u_curr/rho_pred
        v_curr = rho_v_curr/rho_pred

        # Update state
        state['rho']['curr']      = state['rho']['curr'].at[1:-1, 1:-1].set(rho_pred)
        state['vel']['curr']['u'] = state['vel']['curr']['u'].at[1:-1, 1:-1].set(u_curr)
        state['vel']['curr']['v'] = state['vel']['curr']['u'].at[1:-1, 1:-1].set(v_curr)

        # Set boundary condition
        state['vel']['curr'] = self.jit_set_boundary(state['vel']['curr'])

        return state


    def ibcorrector(self, state): 
        return state


    def isdone(self, state):
        state['done'] = state['t'] >= self.config['stop']['time'] 
        return state 


    def display(self, state):
        n = state['n']
        t = state['t']
        #print(f'n: {n} , t: {t:0.6f}')


    def save(self,state):
        n = state['n']
        if n % self.config['save']['nstep'] == 0:
            self.save_cnt += 1
            npads = self.config['save']['npads']
            filename = f'data_{self.save_cnt:0{npads}d}.pkl'
            filepath = pathlib.Path(self.save_dir, filename)
            print(f'saving: {filepath}')
            with open(filepath, 'wb') as f:
                data = {'config': self.config, 'state': state}
                pickle.dump(data, f)
                

    def setup_save(self):
        self.save_dir = pathlib.Path(self.config['save']['directory']).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_cnt = 0


    def load_config(self, config):
        return config if isinstance(config, Config) else Config(filename=config)

    
# --------------------------------------------------------------------------------
@jit
def equilib(rho, u, v, e, w, a0, a1, a2):
    ex, ey = e
    b0 = ex*u + ey*v
    b1 = b0*b0
    b2 = u*u + v*v 
    return rho*w*(1.0 + a0*b0 + a1*b1 - a2*b2)

    





        
