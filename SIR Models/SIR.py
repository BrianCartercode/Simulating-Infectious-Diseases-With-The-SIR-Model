import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SIR model equations
def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


## initial conditions
S0 = 0.99   ##total must add up to 100% 
I0 = 0.01
R0 = 0.0
y0 = [S0, I0, R0]


if S0 + I0 + R0 != 1.0:
    raise ValueError("Initial conditions must sum to 1.0 (100%)")


# Parameters
# β (beta) is transmission rate and I chose 30%. γ (gamma) is set to 1%
beta = 0.3
gamma = 0.1

# Time vector
t = np.linspace(0, 200, 200)  # Simulate for 200 days

# Solve the SIR model equations using odeint()
solution = odeint(SIR_model, y0, t, args=(beta, gamma))

# Extract results
S, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
