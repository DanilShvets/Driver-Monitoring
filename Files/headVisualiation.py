import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

Model3D = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -160.0, -45.0),  # Chin
            (-95.0, 45.0, -90.0),  # Left eye left corner
            (95.0, 45.0, -90.0),  # Right eye right corner
            (-55.0, -80.0, -60.0),  # Left Mouth corner
            (55.0, -80.0, -60.0)  # Right mouth corner

        ])

fig = plt.figure()
ax = plt.axes(projection="3d")

x, y, z = Model3D[:, 0], Model3D[:, 1], Model3D[:, 2]

ax.scatter3D(x, y, z)
plt.show()
