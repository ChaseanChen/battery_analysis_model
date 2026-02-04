import numpy as np
from dae_comparison_A_vs_C_MAIN import run_simulation


# ======================================================
# Slow-time aging laws (cycle-indexed, quasi-static)
# ======================================================

def aging_law_capacity(Q0, k_Q, N):
    """
    Capacity fade: SEI-driven sqrt(N) degradation law
    Q(N) = Q0 * (1 - k_Q * sqrt(N))
    """
    Q = Q0 * (1.0 - k_Q * np.sqrt(N))

    # physical lower bound to avoid nonphysical negative capacity
    Q_min = 0.3 * Q0
    return max(Q, Q_min)


def aging_law_resistance(R0_base, k_R, N):
    """
    Resistance growth: linear-in-cycle approximation
    R0(N) = R0_base * (1 + k_R * N)
    """
    return R0_base * (1.0 + k_R * N)


# ======================================================
# Cycle-level aging runner
# ======================================================

def run_cycle_aging(
    N_cycles=300,
    Qmax0=4.0 * 3600.0,   # [C]
    R0_base=0.12,         # [Ohm]
    k_Q=0.025,            # sqrt-cycle capacity fade coefficient
    k_R=1.5e-3,           # per-cycle resistance growth coefficient
    enable_control=True,
):
    """
    Multi-cycle simulation with slow-time aging and fast-time DAE.

    Each cycle:
      1) Update aging parameters (Qmax, R0)
      2) Treat them as quasi-static constants
      3) Run fast-time Index-1 DAE to obtain TTE
    """

    records = []

    for k in range(1, N_cycles + 1):

        # -------- slow-time aging update --------
        Qmax_k = aging_law_capacity(Qmax0, k_Q, k)
        R0_k   = aging_law_resistance(R0_base, k_R, k)

        # -------- fast-time DAE simulation --------
        result = run_simulation(
            Qmax=Qmax_k,
            R0=R0_k,
            N_cyc=k,
            enable_control=enable_control,
        )

        records.append({
            "cycle": k,
            "TTE": result["TTE"],
            "SOC_crit": result.get("SOC_crit"),
            "shutdown_mode": result.get("shutdown_mode"),
            "Delta_end": result.get("Delta_end"),
            "Qmax": Qmax_k,
            "R0": R0_k,
        })

        # -------- end-of-life early stop --------
        if Qmax_k <= 0.35 * Qmax0:
            print(f"[EOL] Capacity below threshold at cycle {k}")
            break

    return records

if __name__ == "__main__":

    records = run_cycle_aging(N_cycles=50)

    for r in records[:5]:
        print(r)

    print("...")
    print(f"Total cycles simulated: {len(records)}")
