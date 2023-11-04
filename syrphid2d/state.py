import jax.numpy as jnp

def initialize_state(config, lattice):
    state = {}

    # Iteration information
    state['n'] = 0
    state['t'] = 0.0 
    state['done'] = False

    # Mesh grid spacing and x,y extents
    ds = config['mesh']['ds']
    len_x = config['mesh']['len']['x']
    len_y = config['mesh']['len']['y']
    num_x = int(round(len_x/ds))+1
    num_y = int(round(len_y/ds))+1
    density = config['fluid']['density']
    
    # Compute relaxation parameter tau
    kvisc = config['fluid']['kvisc']
    cs = lattice.cs
    state['tau'] = 0.5 + kvisc/(ds*cs**2)

    # Create x,y positions mesh
    x = jnp.linspace(0, len_x, num_x) 
    y = jnp.linspace(0, len_y, num_y)
    grid_x, grid_y = jnp.meshgrid(x, y, indexing='ij')
    state['num_x'] = num_x
    state['num_y'] = num_y

    # Create meshes for density
    state['rho'] = { 
            'last' : jnp.full((num_x, num_y), density),
            'pred' : jnp.full((num_x, num_y), density),
            'curr' : jnp.full((num_x, num_y), density),
            }

    # Create meshes for velocities 
    state['vel'] = {
            'last' : create_velocity_mesh(num_x, num_y, config),
            'pred' : create_velocity_mesh(num_x, num_y, config),
            'curr' : create_velocity_mesh(num_x, num_y, config),
            }
    return state


def create_velocity_mesh(num_x, num_y, config): 
    init_type = config['initial']['type']
    if init_type == 'constant':
        vel_x = config['initial']['velocity']['x']
        vel_y = config['initial']['velocity']['y']
        u = vel_x*jnp.ones((num_x, num_y))
        v = vel_y*jnp.ones((num_x, num_y))
    else:
        raise RuntimeError(f"init type = {init_type} unknown")
    return {'u': u, 'v': v}



#def create_masks(num_x, num_y):
#    # Mesh indices
#    indx, indy = jnp.meshgrid(
#            jnp.arange(num_x, dtype=jnp.uint32), 
#            jnp.arange(num_y, dtype=jnp.uint32), 
#            indexing='ij',
#            )
#
#    # Create masks for boundaries
#    boundary_upper = (indy == (num_y-1))
#    boundary_lower = (indy == 0)
#    boundary_left  = (indx == 0)
#    boundary_right = (indx == (num_x-1))
#    boundary_all = jnp.logical_or(
#            jnp.logical_or(boundary_upper, boundary_lower), 
#            jnp.logical_or(boundary_left,  boundary_right),
#            )
#
#    # Create mask for interior points
#    interior = jnp.logical_not(boundary_all)
#    masks = { 
#            'interior': interior,
#            'boundary': {
#                'all'   : boundary_all,
#                'upper' : boundary_upper, 
#                'lower' : boundary_lower,
#                'left'  : boundary_left, 
#                'right' : boundary_right,
#                }
#            }
#    return masks
