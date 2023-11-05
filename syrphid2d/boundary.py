from jax import jit

class Boundary:

    def __init__(self, config):
        self.left  = LeftBoundarySide(config)
        self.right = RightBoundarySide(config)
        self.upper = UpperBoundarySide(config)
        self.lower = LowerBoundarySide(config)
        self.sides = [self.left, self.right, self.upper, self.lower]
        self.sort_sides()
        self.jit_set = jit(self.set)
        self.jit_set_vel = jit(self.set_vel)
        self.jit_set_rho = jit(self.set_rho)

    def sort_sides(self):
        type_to_sort_key = { 
                'inflow'  : 0, 
                'outflow' : 1, 
                'noslip'  : 2, 
                'moving'  : 3, 
                'slip'    : 4, 
                }
        sort_key_func = lambda side : type_to_sort_key[side.type]
        self.sides.sort(key=sort_key_func)


    def set(self, state):
        for name in ['last', 'pred', 'curr']:
            state[name]['rho'] = self.set_rho(state[name]['rho'])
            state[name]['vel'] = self.jit_set_vel(state[name]['vel'])
        return state


    def set_rho(self, rho):
        c0 = 4.0/3.0
        c1 = 1.0/3.0
        rho = rho.at[ 0, :].set(c0*rho[ 1, :] - c1*rho[ 2, :])
        rho = rho.at[-1, :].set(c0*rho[-2, :] - c1*rho[-3, :])
        rho = rho.at[ :, 0].set(c0*rho[ :, 1] - c1*rho[ :, 2])
        rho = rho.at[ :,-1].set(c0*rho[ :,-2] - c1*rho[ :,-3])
        return rho


    def set_vel(self, vel):
        for side in self.sides:
            vel['u'], vel['v'] = side.set(vel['u'], vel['v'])
        return vel
            

class BoundarySide:
    """ Base class for boundry sides """

    def __init__(self, config, side):
        self.type_to_func = {
                'slip'    : jit(self.set_slip),
                'noslip'  : jit(self.set_noslip),
                'moving'  : jit(self.set_moving),
                'inflow'  : jit(self.set_inflow),
                'outflow' : jit(self.set_outflow),
                }
        self.name = side
        self.type = config['boundary'][side]['type']
        self.set = self.type_to_func[self.type]
        self.value = config['boundary'][side]['value']

    def set_slip(self, u, v):
        pass

    def set_noslip(self, u, v):
        pass

    def set_moving(self, u, v):
        pass

    def set_inflow(self, u, v):
        pass

    def set_outflow(self, u, v):
        pass


class LeftBoundarySide(BoundarySide):

    def __init__(self, config):
        super().__init__(config, 'left')

    def set_slip(self, u, v):
        u = u.at[0,:].set(0.0)
        v = u.y.at[0,:].set(v[1,:])
        return u, v

    def set_noslip(self, u, v):
        u = u.at[0,:].set(0.0)
        v = v.at[0,:].set(0.0)
        return u, v

    def set_moving(self, u, v):
        u = u.at[0,:].set(0.0)
        v = v.at[0,:].set(self.value)
        return u, v

    def set_inflow(self, u, v):
        u = u.at[0,:].set(self.value)
        v = v.at[0,:].set(0.0)
        return u, v

    def set_outflow(self, u, v):
        u = u.at[0,:].set(u[1,:])
        v = v.at[0,:].set(v[1,:])
        return u, v


class RightBoundarySide(BoundarySide):

    def __init__(self, config):
        super().__init__(config, 'right')

    def set_slip(self, u, v):
        u = u.at[-1,:].set(0.0)
        v = v.at[-1,:].set(v[-2,:])
        return u, v

    def set_noslip(self, u, v):
        u = u.at[-1,:].set(0.0)
        v = v.at[-1,:].set(0.0)
        return u, v

    def set_moving(self, u, v):
        u = u.at[-1,:].set(0.0)
        v = v.at[-1,:].set(self.value)
        return u, v

    def set_inflow(self, u, v):
        u = u.at[-1,:].set(self.value)
        v = v.at[-1,:].set(0.0)
        return u, v

    def set_outflow(self, u, v):
        u = u.at[-1,:].set(u[-2,:])
        v = v.at[-1,:].set(v[-2,:])
        return u, v


class UpperBoundarySide(BoundarySide):

    def __init__(self, config):
        super().__init__(config, 'upper')

    def set_slip(self, u, v):
        u = u.at[:,-1].set(u[:,-2])
        v = v.at[:,-1].set(0.0)
        return u, v

    def set_noslip(self, u, v):
        u = u.at[:,-1].set(0.0)
        v = v.at[:,-1].set(0.0)
        return u, v

    def set_moving(self, u, v):
        u = u.at[:,-1].set(self.value)
        v = v.at[:,-1].set(0.0)
        return u, v

    def set_inflow(self, u, v):
        u = u.at[:,-1].set(0.0)
        v = v.at[:,-1].set(self.value)
        return u, v

    def set_outflow(self, u, v):
        u = u.at[:,-1].set(u[:,-2])
        v = v.at[:,-1].set(v[:,-2])
        return u, v


class LowerBoundarySide(BoundarySide):

    def __init__(self, config):
        super().__init__(config, 'lower')

    def set_slip(self, u, v):
        u = u.at[:,0].set(u[:,1])
        v = v.at[:,0].set(0.0)
        return u, v

    def set_noslip(self, u, v):
        u = u.at[:,0].set(0.0)
        v = v.at[:,0].set(0.0)
        return u, v

    def set_moving(self, u, v):
        u = u.at[:,0].set(self.value)
        v = v.at[:,0].set(0.0)
        return u, v

    def set_inflow(self, u, v):
        u = u.at[:,0].set(0.0)
        v = v.at[:,0].set(self.value)
        return u, v

    def set_outflow(self, u, v):
        u = u.at[:,0].set(u[:,1])
        v = v.at[:,0].set(v[:,1])
        return u, v
