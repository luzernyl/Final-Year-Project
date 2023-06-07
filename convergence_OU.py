import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the Ornstein-Uhlenbeck process
def ornstein_uhlenbeck(x0, theta, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        x[i+1] = x[i] + theta * (mu - x[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return x

# Define a reference SDE solver using the Euler-Maruyama method
def EM(x0, theta, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        x[i+1] = x[i] + theta * (mu - x[i]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return x

def SRK1(x0, theta, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        x_hat = x[i] + theta * (mu - x[i]) * dt + sigma * np.sqrt(dt)
        x[i+1] = x[i] + theta * (mu - x_hat) * dt
    return x

def SRK2(x0, theta, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        x_hat = x[i] + theta * (mu - x[i]) * dt + sigma * dt
        f1 = theta * (mu - x[i])
        f2 = theta * (mu - x_hat)
        x[i+1] = x[i] + 0.5 * (f1 + f2) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return x

def SIE(x0, theta, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        s = 1 if np.random.random() < 0.5 else -1
        dw = np.sqrt(dt) * np.random.normal()
        k1 = theta * (mu - x[i]) * dt + (dw - s * np.sqrt(dt)) * sigma
        k2 = theta * (mu - x[i] + k1) * dt + (dw + s * np.sqrt(dt)) * sigma
        x[i+1] = x[i] + 0.5 * (k1 + k2)
    return x

def WeakOrder2RK(x0, theta, mu, sigma, dt, n):
    x = np.zeros(n+1)
    x[0] = x0
    for i in range(n):
        dw = np.sqrt(dt) * np.random.normal()
        f = theta * (mu - x[i])
        x_tilde = x[i] + f * dt + sigma * dw
        x_tilde_pos = x[i] + f * dt + sigma * np.sqrt(dt)
        x_tilde_min = x[i] + f * dt - sigma * np.sqrt(dt)
        f_tilde = theta * (mu - x_tilde)
        f_tilde_pos = theta * (mu - x_tilde_pos)
        f_tilde_min = theta * (mu - x_tilde_min)
        x[i+1] = x[i] + 0.5 * (f + f_tilde) * dt + sigma * dw
    return x

# Define a function to compute the mean and variance of the difference between two solutions
def compare_solutions(ref, sol):
    diff = ref - sol
    mean_diff = np.mean(diff, axis=0)
    var_diff = np.var(diff, axis=0)
    return mean_diff, var_diff

# Set the parameters for the Ornstein-Uhlenbeck process
x0 = 0
theta = 1
mu = 0
sigma = 1

# Set the number of time steps and sample paths
n = 1000
m_values = [10, 50, 100, 500, 1000]

# Generate sample paths using the reference solver
ref_solutions = []
for m in m_values:
    dt = 1.0 / m
    ref_solutions.append(np.array([ornstein_uhlenbeck(x0, theta, mu, sigma, dt, n) for _ in tqdm(range(m))]))

# Generate sample paths using your self-written solver
sol_solutions = []
for m in m_values:
    dt = 1.0 / m
    sol_solutions.append(np.array([EM(x0, theta, mu, sigma, dt, n) for _ in tqdm(range(m))]))

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
fig.tight_layout(pad=3.0)
fig.suptitle("Convergence of EM Solver", fontsize=16)
fig.subplots_adjust(top=0.88)
plt.show()
