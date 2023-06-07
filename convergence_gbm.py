import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# Define the exact solution of Geometric Brownian Motion
def GBM(x0, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    w = 0
    for i in range(n):
        t = i * dt
        w = w + np.sqrt(dt) * np.random.normal()
        x[i+1] = x[0] * np.exp((mu - np.square(sigma) / 2) * t + sigma * w)
    return x

# Define a reference SDE solver using the Euler-Maruyama method
def EM(x0, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        dw = np.sqrt(dt) * np.random.normal()
        x[i+1] = x[i] + mu * x[i] * dt + sigma * x[i] * dw
    return x

def SRK1(x0, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        dw = np.sqrt(dt) * np.random.normal()
        x_hat = x[i] + mu * x[i] * dt + sigma * x[i] * np.sqrt(dt)
        x[i+1] = x[i] + mu * x[i] * dt + 0.5 / np.sqrt(dt) * sigma * (x_hat - x[i]) * (np.square(dw) - dt)
    return x

def SRK2(x0, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        x_hat = x[i] + mu * x[i] * dt + sigma * x[i] * dt
        f1 = mu * x[i]
        f2 = mu * x_hat
        x[i+1] = x[i] + 0.5 * (f1 + f2) * dt + sigma * x[i] * np.sqrt(dt) * np.random.normal()
    return x

def SIE(x0, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        s = 1 if np.random.random() < 0.5 else -1
        dw = np.sqrt(dt) * np.random.normal()
        k1 = mu * x[i] * dt + (dw - s * np.sqrt(dt)) * sigma * x[i]
        k2 = mu * (x[i] + k1) * dt + (dw + s * np.sqrt(dt)) * sigma * (x[i] + k1)
        x[i+1] = x[i] + 0.5 * (k1 + k2)
    return x

def WeakOrder2RK(x0, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        dw = np.sqrt(dt) * np.random.normal()
        f = mu * x[i]
        L = sigma * x[i]
        x_tilde = x[i] + f * dt + L * dw
        x_tilde_pos = x[i] + f * dt + L * np.sqrt(dt)
        x_tilde_min = x[i] + f * dt - L * np.sqrt(dt)
        f_tilde = mu * x_tilde
        f_tilde_pos = mu * x_tilde_pos
        f_tilde_min = mu * x_tilde_min
        L_tilde = sigma * x_tilde
        L_tilde_pos = sigma * x_tilde_pos
        L_tilde_min = sigma * x_tilde_min
        x[i+1] = x[i] + 0.5 * (f + f_tilde) * dt + 0.25 * (L_tilde_pos + 2 * L + L_tilde_min) * dw \
            + 0.25 / np.sqrt(dt) * (L_tilde_pos - L_tilde_min) * (np.square(dw) - dt)
    return x

# Define a function to compute the mean and variance of the difference between two solutions
def compare_solutions(ref, sol):
    diff = ref - sol
    mean_diff = np.mean(diff, axis=0)
    var_diff = np.var(diff, axis=0)
    return mean_diff, var_diff

# Set the parameters for the GBM
x0 = 1
mu = 0.05
sigma = 0.2

# Set the number of time steps and sample paths
n = 1000
m_values = [10, 50, 100, 500, 1000]

# Generate sample paths using the reference solver
ref_solutions = []
print("Exact Solution")
for m in m_values:
    dt = 1.0 / m
    ref_solutions.append(np.array([GBM(x0, mu, sigma, dt, n) for _ in tqdm(range(m))]))

# Generate sample paths using your self-written solver
sol_solutions = []
print("WeakOrder2RK")
for m in m_values:
    dt = 1.0 / m
    sol_solutions.append(np.array([WeakOrder2RK(x0, mu, sigma, dt, n) for _ in tqdm(range(m))]))

# Compute the mean and variance of the difference between the two solutions
mean_diffs = []
var_diffs = []
for i in range(len(m_values)):
    mean_diff, var_diff = compare_solutions(ref_solutions[i], sol_solutions[i])
    mean_diffs.append(mean_diff)
    var_diffs.append(var_diff)

# Plot the mean and variance of the difference as a function of time
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
for i in range(len(m_values)):
    ax[0].plot(mean_diffs[i], label=f"m={m_values[i]}")
    ax[1].plot(var_diffs[i], label=f"m={m_values[i]}")
ax[0].set_title("Mean Difference vs Time")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Mean Difference")
ax[0].legend()
ax[1].set_title("Variance of Difference vs Time")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("Variance of Difference")
ax[1].legend()
fig.tight_layout(pad = 3.0)
fig.suptitle("Convergence of WeakOrder2RK Solver", fontsize=16)
fig.subplots_adjust(top=0.88)
plt.show()
