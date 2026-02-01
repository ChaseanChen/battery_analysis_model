import numpy as np
import matplotlib.pyplot as plt

# ---------- Physical Models ----------

def get_ocv(soc):
    """
    Empirical OCV model.
    SOC is clipped to avoid non-physical divergence near 0 and 1.
    """
    soc = np.clip(soc, 0.01, 0.99)
    return 3.5 + 0.45*soc - 0.04/soc - 0.08*np.log(1 - soc)

def get_resistance(T_celsius, N_cyc):
    """
    Temperature- and aging-dependent internal resistance.
    Arrhenius term is exponent-limited for numerical stability.
    """
    R_base = 0.12
    beta = 8e-4
    Ea = 2e4
    Rg = 8.314

    T_k = T_celsius + 273.15
    T_ref = 298.15

    R_aging = 1.0 + beta * N_cyc
    exponent = Ea / Rg * (1.0 / T_k - 1.0 / T_ref)
    exponent = np.clip(exponent, -8.0, 8.0)  # numerical safety

    return R_base * R_aging * np.exp(exponent)

# ---------- Discriminant ----------

def calc_discriminant(soc, T, N, P_load):
    """
    Δ = V_ocv^2 - 4 R P
    """
    V = get_ocv(soc)
    R = get_resistance(T, N)
    return V**2 - 4.0 * R * P_load

# ---------- Main ----------

if __name__ == "__main__":
    soc = np.linspace(0.01, 1.0, 500)
    P_req = 8.0  # W

    delta_new = calc_discriminant(soc, T=25, N=0, P_load=P_req)
    delta_old = calc_discriminant(soc, T=0, N=600, P_load=P_req)

    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(soc*100, delta_new, lw=2, label="New Cell (25°C)")
    plt.plot(soc*100, delta_old, lw=2, label="Aged Cell (0°C)")

    plt.axhline(0, ls="--", color="k", alpha=0.6)
    plt.fill_between(soc*100, -5, 0, color="gray", alpha=0.2,
                    label="Infeasible Region")

    plt.xlim(0, 100)
    plt.ylim(-2, 12)
    plt.xlabel("SOC (%)")
    plt.ylabel(r"Discriminant $\Delta$")
    plt.title(f"Power Feasibility Boundary ({P_req} W Load)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Fig5.1_Power_Feasibility_Boundary_P8W.png", dpi=300)
    plt.show()
