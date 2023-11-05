import time
import json
import pickle
import pathlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
from  matplotlib.animation import FuncAnimation
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
        self.jit_set_boundaries = jit(self.set_boundaries)
        self.jit_predictor = jit(self.predictor)
        self.jit_corrector = jit(self.corrector)
        self.jit_ibcorrector = jit(self.ibcorrector)
        self.jit_isdone = jit(self.isdone)


    def run(self):
        self.setup_save()
        state = initialize_state(self.config, self.lattice)
        state = self.jit_set_boundaries(state)
        self.setup_animation(state)
        while not state['done']:
            state = self.jit_update_iter(state)
            state = self.jit_predictor(state)
            state = self.jit_corrector(state)
            state = self.jit_ibcorrector(state)
            state = self.jit_isdone(state)
            self.save(state)
            self.update_animation(state)


    def set_boundaries(self,state):
        state = self.boundary.jit_set(state)
        return state


    def update_iter(self,state):
        state['n'] += 1
        state['t'] += self.config['mesh']['ds']
        return state


    def display_iter(self, state):
        n = state['n']
        t = state['t']
        print(f'n: {n} , t: {t:0.6f}')


    def predictor(self, state):

        num_x, num_y = state['last']['rho'].shape
        avals = self.lattice.avals

        # Set last state to current state 
        state['last']['rho']      = state['curr']['rho']
        state['last']['vel']['u'] = state['curr']['vel']['u']
        state['last']['vel']['v'] = state['curr']['vel']['v']

        # Initialize rho, rho*u, and rho*v matricies for predictor step
        rho_pred     = jnp.zeros((num_x-2, num_y-2))
        rho_u_pred   = jnp.zeros((num_x-2, num_y-2))
        rho_v_pred   = jnp.zeros((num_x-2, num_y-2))

        for e, w in zip(self.lattice.e, self.lattice.w):

            # Slice indices for equilibrium calculation
            ex, ey = e           # lattice basis vectors
            nx = 1 - ex          # slice start x index
            ny = 1 - ey          # slice start y index
            mx = num_x + nx - 2  # slice end x index
            my = num_y + ny - 2  # slice end y index

            # Get slices of current state for equilibrium calculation
            rho_curr_slice = state['curr']['rho'][nx:mx, ny:my] 
            u_curr_slice   = state['curr']['vel']['u'][nx:mx, ny:my]
            v_curr_slice   = state['curr']['vel']['v'][nx:mx, ny:my]

            # Compute equilibrium function and update predictor rho, u and v  
            feq = equilib(rho_curr_slice, u_curr_slice, v_curr_slice, e, w, *avals) 
            rho_pred   += feq
            rho_u_pred += feq*ex
            rho_v_pred += feq*ey

        u_pred = rho_u_pred/rho_pred
        v_pred = rho_v_pred/rho_pred

        # Update state 
        state['pred']['rho']      = state['pred']['rho'].at[1:-1,1:-1].set(rho_pred)
        state['pred']['vel']['u'] = state['pred']['vel']['u'].at[1:-1,1:-1].set(u_pred)
        state['pred']['vel']['v'] = state['pred']['vel']['v'].at[1:-1,1:-1].set(v_pred)

        # Set boundary condition 
        state['pred']['vel'] = self.boundary.jit_set_vel(state['pred']['vel'])
        state['pred']['rho'] = self.boundary.jit_set_rho(state['pred']['rho'])
        return state


    def corrector(self, state):

        num_x, num_y = state['last']['rho'].shape
        avals = self.lattice.avals
        tau = state['tau']

        # Get interior slices for predictor velocity 
        rho_pred = state['pred']['rho'][1:-1, 1:-1]
        u_pred   = state['pred']['vel']['u'][1:-1, 1:-1]
        v_pred   = state['pred']['vel']['v'][1:-1, 1:-1]

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
            rho_pred_slice = state['pred']['rho'][nx:mx, ny:my] 
            u_pred_slice   = state['pred']['vel']['u'][nx:mx, ny:my] 
            v_pred_slice   = state['pred']['vel']['v'][nx:mx, ny:my] 

            # Compute equilibrium function and update rho*u and rho*v
            feq = equilib(rho_pred_slice, u_pred_slice, v_pred_slice, e, w, *avals) 
            rho_u_curr += (tau - 1)*feq*ex
            rho_v_curr += (tau - 1)*feq*ey

        # Get interior slices for rho, u and v last
        rho_last = state['last']['rho'][1:-1, 1:-1]
        u_last   = state['last']['vel']['u'][1:-1, 1:-1]
        v_last   = state['last']['vel']['v'][1:-1, 1:-1]

        # Update rho*u and rho*v current and divide by rho to get u and v
        rho_u_curr -= (tau - 1)*rho_last*u_last
        rho_v_curr -= (tau - 1)*rho_last*v_last
        u_curr = rho_u_curr/rho_pred
        v_curr = rho_v_curr/rho_pred

        # Update state
        state['curr']['rho']      = state['curr']['rho'].at[1:-1, 1:-1].set(rho_pred)
        state['curr']['vel']['u'] = state['curr']['vel']['u'].at[1:-1, 1:-1].set(u_curr)
        state['curr']['vel']['v'] = state['curr']['vel']['v'].at[1:-1, 1:-1].set(v_curr)

        # Set boundary condition
        state['curr']['rho'] = self.boundary.jit_set_rho(state['curr']['rho'])
        state['curr']['vel'] = self.boundary.jit_set_vel(state['curr']['vel'])
        return state


    def ibcorrector(self, state): 
        return state


    def isdone(self, state):
        state['done'] = state['t'] >= self.config['stop']['time'] 
        return state 


    def save(self,state):
        if state['n'] % self.config['save']['nstep'] == 0:
            self.save_cnt += 1
            npads = self.config['save']['npads']
            filename = f'data_{self.save_cnt:0{npads}d}.pkl'
            filepath = pathlib.Path(self.save_dir, filename)
            print()
            print(f"n: {state['n']:08d}, t: {state['t']:0.2f}s")
            print(f'saving: {filepath}')
            with open(filepath, 'wb') as f:
                state_to_save =  {k:v for k,v in state.items() if not k in ['last', 'pred']}
                data = {'config' : self.config.to_dict(), 'state'  : state_to_save}
                pickle.dump(data, f)
                

    def setup_save(self):
        self.save_dir = pathlib.Path(self.config['save']['directory']).expanduser()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_cnt = 0


    def plot(self, state, animation=False):
        x, y = jnp.meshgrid(state['x'], state['y'], indexing='ij')
        n = state['n']
        t = state['t']
        u = state['curr']['vel']['u']
        v = state['curr']['vel']['v']

        fig, ax = plt.subplots(1,1, figsize=(10,9))
        quiver = plt.quiver(x, y, u, v, scale=3.0)
        ax.set_title(f'n: {n:06d}, t: {t:8.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        if animation:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
        else:
            plt.show()
        return fig, ax, quiver 


    def setup_animation(self, state):
        if self.config['plots']['animation']['enabled']:
            plt.ion()
            self.anim_fig, self.anim_ax, self.anim_quiver = self.plot(state)


    def update_animation(self, state):
        if self.config['plots']['animation']['enabled']:
            if state['n'] % self.config['plots']['animation']['nstep'] == 0:
                n = state['n']
                t = state['t']
                u = state['curr']['vel']['u']
                v = state['curr']['vel']['v']
                self.anim_ax.set_title(f'n: {n:06d}, t: {t:8.2f}s')
                self.anim_quiver.set_UVC(u,v)
                self.anim_fig.canvas.draw()
                self.anim_fig.canvas.flush_events()
                plt.pause(0.001)


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

    





        
