#to create the environment (not needed anymore): python -m venv myenv
#to activate the enviro: myenv\Scripts\activate
#to load the modules used: pip install -r requirements.txt
 
import matplotlib.pyplot as plt
import numpy as np

import random

alpha_show = 0.9

pi = 0.99
alpha_min = 2/3/pi
gamma = 1e-3
density = 1000

def g0(x, alpha):
    return x*x*x + 3*x*x*(1-x)*alpha*(1-pi) + 3*x*(1-x)*(1-x)*(alpha*(1-pi) + 1-alpha)

def g1(x, alpha):
    return x*x*x + 3*x*x*(1-x)*alpha*pi + 3*x*(1-x)*(1-x)*(alpha*pi + 1-alpha)

def psi1(x):
    return x*(1-x)*(1-x)

def psi2(x):
    return x*x*(1-x)


epsilon = 1e-3
a = 1e-10
K=100

I1s = []
I2s = []
ps = []
alphas = []
alls_ys = []
for d in range(density):
    alpha = alpha_min + gamma + d*(1-alpha_min- gamma)/density
    I1 = 0
    I2 = 0

    x0 = a
    x1 = g1(x0, alpha)
    logys = np.linspace(np.log(x0), np.log(x1), K+1)
    ys = np.exp(logys)
    all_ys = ys
    
    while x1 < 1 - epsilon:
        x0 = x1
        x1 = g1(x0, alpha)
        ys = g1(ys, alpha)
        all_ys = np.concatenate((all_ys, ys))
        I1 += 1/K*np.sum(psi1(ys))
        I2 += 1/K*np.sum(psi2(ys))
    
    alphas.append(alpha)
    I1s.append(I1)
    I2s.append(I2)
    alls_ys.append(all_ys)
    ps.append(I1/(I1+I2))

    if alpha<alpha_show:
        show = [alpha, I1/(I1+I2), d]

# Create a labeled graph
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Plot your data in the first subplot
axs[0].plot(alphas, ps, linestyle='-')
axs[0].plot(show[0], show[1], marker = 'o', color = 'red')
axs[0].set_xlabel('Alphas')
axs[0].set_ylabel('Ps')
axs[0].set_title('Labeled Graph')
axs[0].grid(True)

xs = np.linspace(0, 1, 1000)
y0s = g0(xs, show[0])
y1s = g1(xs, show[0])
axs[1].plot(y0s, xs, linestyle='-', color = 'green')
axs[1].plot(y1s, xs, linestyle='-', color = 'blue')
axs[1].set_xlabel('g_0 and g_1 for alpha=1')
axs[1].set_ylabel('Ps')
axs[1].set_title('Labeled Graph')
axs[1].grid(True)

all_ys = alls_ys[show[2]]
f1s = all_ys
f0s = 1-all_ys
axs[2].hist(f0s, bins=100, color='green', alpha=0.7)
axs[2].hist(f1s, bins=100, color='blue', alpha=0.7)
axs[2].set_xlabel('Values')
axs[2].set_ylabel('Frequency')
axs[2].set_title('Histogram')
axs[2].grid(True)

# Display the plot
plt.tight_layout()
plt.show()