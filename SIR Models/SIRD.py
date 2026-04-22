import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. Define the SIRD model differential equations
def sird_model(y, t, N, beta, gamma, mu):
    S, I, R, D = y
    # Change in Susceptible
    dSdt = -beta * S * I / N
    # Change in Infected
    dIdt = (beta * S * I / N) - (gamma * I) - (mu * I)
    # Change in Recovered
    dRdt = gamma * I
    # Change in Deceased
    dDdt = mu * I
    return dSdt, dIdt, dRdt, dDdt

# 2. Parameters
N = 10000        # Total population
I0 = 10          # Initial infected
R0 = 0           # Initial recovered
D0 = 0           # Initial deceased
S0 = N - I0 - R0 - D0 # Initial susceptible

beta = 0.5       # Infection rate (contact * transmission probability)
gamma = 0.1      # Recovery rate (1 / days to recover)
mu = 0.01        # Mortality rate (1 / days to die)

# Time points (in days)
t = np.linspace(0, 150, 150)

# 3. Solve the differential equations
y0 = (S0, I0, R0, D0)
solution = odeint(sird_model, y0, t, args=(N, beta, gamma, mu))
S, I, R, D = solution.T

# 4. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible', color='blue')
plt.plot(t, I, label='Infected', color='orange')
plt.plot(t, R, label='Recovered', color='green')
plt.plot(t, D, label='Deceased', color='red')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIRD Epidemic Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
