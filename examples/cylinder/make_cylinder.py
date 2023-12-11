import toml
import numpy as np
import matplotlib.pyplot as plt

filename = 'cylinder.toml'

# Cylinder parameters
cx = 2.0
cy = 1.0
diam = 0.25
num_pts = 100

# Tunnel parameters
len_x = 8
len_y = 2

# Create cylinder points
angles = np.linspace(0,2*np.pi, num_pts, endpoint=False)
xvals = diam*np.cos(angles) + cx
yvals = diam*np.sin(angles) + cy

# Write .toml file
cylinder_data = {
        'closed' : False,
        'points' : {
            'x' : xvals.tolist(),
            'y' : yvals.tolist(),
            },
        'motion' : 'static',
        }

with open(filename, 'w') as f:
    toml.dump(cylinder_data, f)


# Plot cylider points
fig, ax = plt.subplots(1,1)
ax.plot(xvals, yvals, '.-b')
ax.plot(xvals[:1], yvals[:1], '.g')
ax.plot(xvals[-1:], yvals[-1:], '.r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.axis('scaled')
ax.set_xlim(0,len_x)
ax.set_ylim(0,len_y)
plt.show()



