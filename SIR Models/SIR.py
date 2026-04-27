"""
Title: SIR Model Simulation for Infectious Disease Spread

Author: Brian Carter
Project: Simulating Infectious Diseases Using SIR Models

Description:
This script implements the Susceptible-Infected-Recovered (SIR) model
to simulate the spread of infectious diseases within a population.
The model is solved numerically using the SciPy library and visualised
using Matplotlib. Multiple scenarios are analysed by varying the
transmission rate (β), allowing comparison between weak, mild, and
severe infections. Intervention strategies are also simulated by
modifying the transmission rate over time.

Source:
This implementation is adapted from:
https://hackernoon.com/simulating-infectious-disease-spread-with-python-sir-and-seir-models

Modifications:
- Converted population values to proportions for normalisation
- Added multiple infection scenarios (weak, mild, severe)
- Implemented intervention strategies with time-dependent β
- Calculated key metrics (peak infection, time to peak, total infected)
- Integrated visualisation of results using Matplotlib

Dependencies:
- numpy
- scipy
- matplotlib

Date: [Add date here]
"""


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt



def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


# including intervention in the model
def SIR_intervention_model(y, t, beta, gamma, intervention_day, reduction_factor):
    S, I, R = y

    if t >= intervention_day:
        beta = beta * reduction_factor

    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]


# find key metrics from the results then print them out
def analyse_results(t, S, I, R, beta_values, gamma):
    peak_infected = np.max(I)
    time_to_peak = t[np.argmax(I)]
    final_recovered = R[-1]
    final_susceptible = S[-1]

    # Effective reproduction number
    Rt = (beta_values * S) / gamma

    threshold_indices = np.where(Rt <= 1)[0]

    if len(threshold_indices) > 0:
        rt_day = t[threshold_indices[0]]
    else:
        rt_day = None

    return peak_infected, time_to_peak, final_recovered, final_susceptible, rt_day


#starting conditions
S0 = 0.99
I0 = 0.01
R0 = 0.0
y0 = [S0, I0, R0]

gamma = 0.1
t = np.linspace(0, 200, 200)

scenarios = {
    "Weak infection": 0.15,
    "Mild infection": 0.25,
    "Severe infection": 0.50
}

if S0 + I0 + R0 != 1.0:
    raise ValueError("Initial conditions must sum to 1.0 (100%)")


#baseline simulations without intervention
print("\n--- BASELINE SIR RESULTS ---")

for name, beta in scenarios.items():

    solution = odeint(SIR_model, y0, t, args=(beta, gamma))
    S, I, R = solution.T

    beta_values = np.full_like(t, beta)

    peak, peak_day, final_R, final_S, rt_day = analyse_results(
        t, S, I, R, beta_values, gamma
    )

    print(f"\n{name}")
    print(f"Beta: {beta}")
    print(f"Gamma: {gamma}")
    print(f"R0: {beta/gamma:.2f}")
    print(f"Peak infected: {peak:.2f}")
    print(f"Time to peak: {peak_day:.1f} days")
    print(f"Final recovered: {final_R:.2f}")
    print(f"Final susceptible: {final_S:.2f}")

    if rt_day is not None:
        print(f"Rt drops below 1 at day: {rt_day:.1f}")
    else:
        print("Rt does not fall below 1")

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label="Susceptible")
    plt.plot(t, I, label="Infected")
    plt.plot(t, R, label="Recovered")
    plt.xlabel("Time (days)")
    plt.ylabel("Proportion of Population")
    plt.title(f"SIR Baseline - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()


#simulations with intervention 
print("\n--- SIR WITH INTERVENTION ---")

intervention_day = 30
reduction_factor = 0.5  # 50% reduction

for name, beta in scenarios.items():

    solution = odeint(
        SIR_intervention_model,
        y0,
        t,
        args=(beta, gamma, intervention_day, reduction_factor)
    )

    S, I, R = solution.T

    # beta changes after intervention
    beta_values = np.where(t >= intervention_day, beta * reduction_factor, beta)

    peak, peak_day, final_R, final_S, rt_day = analyse_results(
        t, S, I, R, beta_values, gamma
    )

    print(f"\n{name} (with intervention)")
    print(f"Original beta: {beta}")
    print(f"Reduced beta after day {intervention_day}: {beta * reduction_factor}")
    print(f"Gamma: {gamma}")
    print(f"Peak infected: {peak:.2f}")
    print(f"Time to peak: {peak_day:.1f} days")
    print(f"Final recovered: {final_R:.2f}")
    print(f"Final susceptible: {final_S:.2f}")

    if rt_day is not None:
        print(f"Rt drops below 1 at day: {rt_day:.1f}")
    else:
        print("Rt does not fall below 1")

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label="Susceptible")
    plt.plot(t, I, label="Infected")
    plt.plot(t, R, label="Recovered")
    plt.axvline(intervention_day, linestyle="--", label="Intervention")
    plt.xlabel("Time (days)")
    plt.ylabel("Proportion of Population")
    plt.title(f"SIR Intervention - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()
