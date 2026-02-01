import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.grid': True})

# ---------- Core Models ----------

def R0(Tc, N):
    R_base, beta = 0.12, 8e-4
    Ea, Rg = 2e4, 8.314
    T_ref = 298.15

    T_k = Tc + 273.15
    exponent = Ea / Rg * (1/T_k - 1/T_ref)
    exponent = np.clip(exponent, -8.0, 8.0)

    return R_base * (1 + beta*N) * np.exp(exponent)

def ocv(soc):
    soc = np.clip(soc, 0.01, 0.99)
    return 3.5 + 0.45*soc - 0.04/soc - 0.08*np.log(1 - soc)

# ---------- Figure 5 ----------

def fig5_R0():
    Tc = np.linspace(-10, 45, 50)
    N = np.linspace(0, 800, 50)
    Tc_m, N_m = np.meshgrid(Tc, N)

    R = R0(Tc_m, N_m)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Tc_m, N_m, R, cmap="viridis")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Cycle Number")
    ax.set_zlabel("Internal Resistance (Ω)")
    ax.set_title("Temperature–Aging Coupled Resistance")
    plt.tight_layout()
    plt.savefig("Fig3.4_R0_Temperature_Cycle_Surface.png", dpi=300)

# ---------- Figure 6 ----------

def fig6_throttling():
    V = np.linspace(2.8, 4.2, 400)
    kv, Vc = 4.0, 3.25
    lam = 0.5 * (1 + np.tanh(kv * (V - Vc)))

    plt.figure(figsize=(8, 5))
    plt.plot(V, lam, lw=2)
    plt.axvline(3.0, ls="--", color="r")
    plt.xlabel("Terminal Voltage (V)")
    plt.ylabel("Throttling Factor λ")
    plt.title("Voltage-Based Power Throttling")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("Fig3.2_Voltage_Based_Power_Throttling.png", dpi=300)

# ---------- Figure 7 ----------

def fig7_discriminant():
    soc = np.linspace(0.01, 1.0, 300)
    P = 7.0

    Δ_new = ocv(soc)**2 - 4 * R0(25, 0) * P
    Δ_old = ocv(soc)**2 - 4 * R0(0, 600) * P

    plt.figure(figsize=(8, 5))
    plt.plot(soc*100, Δ_new, label="New Cell")
    plt.plot(soc*100, Δ_old, label="Aged Cell")
    plt.axhline(0, ls="--", color="r")
    plt.fill_between(soc*100, -5, 0, alpha=0.15)
    plt.xlabel("SOC (%)")
    plt.ylabel("Discriminant Δ")
    plt.title("Evolution of Power Feasibility Boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fig4.2_Discriminant_Boundary_vs_SOC.png", dpi=300)

if __name__ == "__main__":
    fig5_R0()
    fig6_throttling()
    fig7_discriminant()
    plt.show()
