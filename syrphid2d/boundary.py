import jax

class Boundary:

    def __init__(self, config):
        self.left  = LeftBoundarySide(config)
        self.right = RightBoundarySide(config)
        self.upper = UpperBoundarySide(config)
        self.lower = LowerBoundarySide(config)
        self.sides = [self.left, self.right, self.upper, self.lower]
        self.sort_sides()

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

    def set(self, vel):
        for side in self.sides:
            vel['u'], vel['v'] = side.set(vel['u'], vel['v'])
        return vel
            

class BoundarySide:
    """ Base class for boundry sides """

    def __init__(self, config, side):
        self.type_to_func = {
                'slip'    : jax.jit(self.set_slip),
                'noslip'  : jax.jit(self.set_noslip),
                'moving'  : jax.jit(self.set_moving),
                'inflow'  : jax.jit(self.set_inflow),
                'outflow' : jax.jit(self.set_outflow),
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
        u = u.at[0,:].set(self.vel.x[1,:])
        v = v.at[0,:].set(self.vel.y[1,:])
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
        u = u.at[-1,:].set(self.vel.x[-2,:])
        v = v.at[-1,:].set(self.vel.y[-2,:])
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
        u = u.at[:,-1].set(self.vel.x[:,-2])
        v = v.at[:,-1].set(self.vel.y[:,-2])
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
        u = u.at[:,0].set(self.vel.x[:,1])
        v = v.at[:,0].set(self.vel.y[:,1])
        return u, v