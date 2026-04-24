import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions
S0 = 0.99
I0 = 0.01
R0 = 0.0
y0 = [S0, I0, R0]

# Fixed recovery rate
gamma = 0.1

# Time period
t = np.linspace(0, 200, 200)

# Different infection scenarios
scenarios = {
    "Weak infection": 0.15,
    "Mild infection": 0.25,
    "Severe infection": 0.50
}

for name, beta in scenarios.items():
    solution = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = solution.T

    peak_infected = max(I)
    time_to_peak = t[np.argmax(I)]
    final_recovered = R[-1]
    final_susceptible = S[-1]
    reproduction_number = beta / gamma

    print(f"\n{name}")
    print(f"Beta: {beta}")
    print(f"Gamma: {gamma}")
    print(f"R0: {reproduction_number:.2f}")
    print(f"Peak infected: {peak_infected:.2f}")
    print(f"Time to peak: {time_to_peak:.1f} days")
    print(f"Final recovered: {final_recovered:.2f}")
    print(f"Final susceptible: {final_susceptible:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of Population')
    plt.title(f'SIR Model Simulation - {name}')
    plt.legend()
    plt.grid(True)
    plt.show()