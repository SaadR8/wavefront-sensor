import numpy as np
import matplotlib.pyplot as plt
from zernike import RZern
import seaborn as sns

sns.set_context('poster')

Phi = np.load("Phi.npy")

cart = RZern(6)
L, K = 720, 720
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)

z = cart.fit_cart_grid(Phi)[0]
#z = np.array([0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print(z)

#pol = RZern(6)
#pol.make_pol_grid(np.linspace(0.0, 1.0), np.linspace(0.0, 2*np.pi))

polPhi = cart.eval_grid(z, matrix=True)


plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(polPhi, origin='lower', extent=(-1, 1, -1, 1))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.plot(range(1, cart.nk + 1), z, marker='.')
plt.show()


#plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
# plt.axis('off')
