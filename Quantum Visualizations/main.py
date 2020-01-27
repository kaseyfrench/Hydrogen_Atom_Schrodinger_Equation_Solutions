# Kasey French
#
#
# Graphical representation of the first few solutions to the Schrodinger Equation
# First we solve for the classic problem of a box in an infinite square well
# Second we solve for solutions for a hydrogen atom's quantum levels

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d



# Plancs constant and mass of electron
h = 6.626e-26
m = 9.11e-31

# Values of x (the infinite square well)
x_list = np.linspace(0,1,100)
# Value for L, the length
L = 1


def psi(n, L, x):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def psi_squared(n, L, x):
    return np.square(psi(n, L, x))



plt.figure(figsize = (15,10))
plt.suptitle("Wave Functions", fontsize = 18)

for n in range(1,4):
    l1 = []
    l2 = []
    for x in x_list:
        l1.append(psi(n, L, x))
        l2.append(psi_squared(n, L, x))

    plt.subplot(3, 2, 2*n - 1)
    plt.plot(x_list, l1)
    plt.xlabel("L", fontsize = 13)
    plt.ylabel("psi", fontsize = 13)
    plt.xticks(np.arange(0,1, step = 0.5))
    plt.title("n = " + str(n), fontsize = 16)
    plt.grid()

    plt.subplot(3, 2, 2*n)
    plt.plot(x_list, l2)
    plt.xlabel("L", fontsize = 13)
    plt.ylabel("psi squared", fontsize = 13)
    plt.xticks(np.arange(0,1, step = 0.5))
    plt.title("n = " + str(n), fontsize = 16)
    plt.grid()

plt.tight_layout()
plt.show()



def prob_1s(x, y, z):
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return np.square(np.exp(-r) / np.sqrt(np.pi))


def prob_2s(x, y, z):
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return np.square((2 - r) * np.exp(-r / 2) / np.sqrt(32 * np.pi))


def prob_210(x, y, z):
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    return np.square(np.cos(np.arctan(np.sqrt(x ** 2 + y ** 2) / z)) * r * np.exp(-r / 2) / np.sqrt(32 * np.pi))

def prob_211(x, y, z):
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    theta = np.arctan(np.sqrt(x ** 2 + y ** 2) / z)
    # phi = np.arctan(x/y)
    # exponent = complex(0, phi)
    return np.square(np.sin(theta) * r * np.exp(-r / 2) / np.sqrt(64 * np.pi))

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)
z = np.linspace(-10,10,100)

elements = []
probability = [[], [], [], []]

for dx in x:
    for dy in y:
        for dz in z:
            elements.append(str((dx,dy,dz)))
            probability[0].append(prob_1s(dx, dy, dz))
            probability[1].append(prob_2s(dx, dy, dz))
            probability[2].append(prob_210(dx, dy, dz))
            probability[3].append(prob_211(dx, dy, dz))

fig = plt.figure(figsize = (20,8))
names = {0: "100 Orbital", 1: "200 Orbital", 2: "210 Orbital", 3: "211 Orbital" }

for p in range(0,4):        
    probability[p] = probability[p] / sum(probability[p])

    coord = np.random.choice(elements, size = 50000, replace = True, p = probability[p])
    elem_mat = [i.split(",") for i in coord]
    elem_mat = np.matrix(elem_mat)
    x_coords = [float(i.item()[1:]) for i in elem_mat[:,0]]
    y_coords = [float(i.item()) for i in elem_mat[:,1]]
    z_coords = [float(i.item()[0:-1]) for i in elem_mat[:,2]]

    ax = fig.add_subplot(2, 4, p + 1,  projection ='3d')
    ax.scatter(x_coords, y_coords, z_coords, alpha = 0.1, s = 2)
    plt.title(names[p])
    plt.gca().set_aspect('equal', adjustable='box')

    ax = fig.add_subplot(2, 4, p + 5)
    ax.scatter(x_coords, y_coords, alpha = 0.1, s = 2)
    plt.title(names[p])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')




#ax.subplot(1,4,1)
#plt.title("100 Shell")
#ax.subplot(1,4,2)
#plt.title("100 Shell")
#ax.subplot(1,4,3)
#plt.title("100 Shell")
#ax.subplot(1,4,4)
#plt.title("100 Shell")

#ax.set_title("Hydrogen Electron Densities")
plt.tight_layout()
plt.show()



