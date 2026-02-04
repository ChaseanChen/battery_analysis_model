"""
SOC asymptotically approaches a residual value under throttling.

In Case C, SOC asymptotically approaches a residual value and the simulation
is truncated at Tmax to avoid excessively long low-power tails.
"""

import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Core battery model
# ======================================================
def ocv(soc):
    return 3.0 + 1.15 * soc - 0.18 * np.exp(-10 * soc)

def lambda_v(V, enable=True, Vcrit=3.4, eps=0.05):
    if not enable:
        return 1.0
    return 0.5 * (1 + np.tanh((V - Vcrit) / eps))

def solve_current(Vest, P, R0):
    Delta = Vest**2 - 4 * R0 * P
    if Delta <= 0:
        return 0.0, Delta
    I = (Vest - np.sqrt(Delta)) / (2 * R0)
    return I, Delta


# ======================================================
# Simulation runner (fast-time DAE)
# ======================================================
def run_simulation(
    enable_control=True,
    Qmax=None,
    R0=None,
    N_cyc=None,
):
    """
    Fast-time DAE simulation.

    If Qmax / R0 are not provided, default values are used
    to preserve original standalone behavior.
    """

    # ---- time step ----
    dt = 0.5

    # ---- default parameters (original behavior) ----
    if Qmax is None:
        Q = 4.0 * 3600.0
    else:
        Q = Qmax

    if R0 is None:
        R0_eff = 0.12
    else:
        R0_eff = R0

    # ---- RC parameters ----
    R1, C1 = 0.02, 2500
    R2, C2 = 0.04, 800

    # ---- initial states ----
    soc = 1.0
    vc1, vc2 = 0.0, 0.0
    t = 0.0

    # ---- records ----
    T, Vt, Delta_hist = [], [], []

    # ---- load ----
    P_base = 8.0
    T_max = 50.0     # hours
    I_min = 1e-4

    shutdown_mode = "unknown"

    while soc > 0 and t / 3600 < T_max:

        Vest = ocv(soc) - vc1 - vc2
        lam = lambda_v(Vest, enable=enable_control)
        P = P_base * lam

        I, Delta = solve_current(Vest, P, R0_eff)

        if Delta <= 0:
            shutdown_mode = "power_collapse"
            break

        if I < I_min:
            shutdown_mode = "throttled_tail"
            break

        soc -= I * dt / Q
        vc1 += dt * (-vc1 / (R1 * C1) + I / C1)
        vc2 += dt * (-vc2 / (R2 * C2) + I / C2)

        V = ocv(soc) - vc1 - vc2 - I * R0_eff

        T.append(t / 3600)
        Vt.append(V)
        Delta_hist.append(Delta)

        t += dt

    if soc <= 0:
        shutdown_mode = "energy_depletion"

    return {
        "TTE": t / 3600,
        "T": np.array(T),
        "V": np.array(Vt),
        "Delta": np.array(Delta_hist),
        "Delta_end": Delta_hist[-1] if len(Delta_hist) > 0 else None,
        "SOC_end": soc,
        "shutdown_mode": shutdown_mode,
        "N_cyc": N_cyc,
    }


# ======================================================
# Standalone run (unchanged behavior)
# ======================================================
if __name__ == "__main__":

    res_A = run_simulation(enable_control=False)
    res_C = run_simulation(enable_control=True)

    TA, VA, DA = res_A["T"], res_A["V"], res_A["Delta"]
    TC, VC, DC = res_C["T"], res_C["V"], res_C["Delta"]

    # ---- Plot: Case A ----
    plt.figure(figsize=(6, 4))
    plt.plot(TA, VA, lw=2, color="tab:blue")
    plt.title("Case A: Open-loop Voltage", fontsize=12)
    plt.ylabel("Voltage (V)", fontsize=10)
    plt.xlabel("Time (h)", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Fig4.3_Voltage_Collapse_OpenLoop.png", dpi=300)
    plt.show()

    # ---- Plot: Case C ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(TC, VC, lw=2, color="tab:green")
    axes[0].set_title("Case C: Controlled Voltage", fontsize=12)
    axes[0].set_ylabel("Voltage (V)", fontsize=10)
    axes[0].set_xlabel("Time (h)", fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(TC, DC, lw=2, color="tab:red")
    axes[1].axhline(0, ls="--", c="black", alpha=0.7)
    axes[1].set_title("Case C: Discriminant $\\Delta$", fontsize=12)
    axes[1].set_ylabel("$\\Delta = V^2 - 4R_0P$", fontsize=10)
    axes[1].set_xlabel("Time (h)", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig4.4_Voltage_and_Discriminant_Controlled.png", dpi=300)
    plt.show()
