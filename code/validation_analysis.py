import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# =========================
# OCV model
# =========================
def ocv(soc):
    return 3.0 + 1.15*soc - 0.18*np.exp(-10*soc)

# =========================
# Power throttling (λ)
# =========================
def lambda_v(V, Vcrit=3.4, eps=0.05):
    return 0.5 * (1 + np.tanh((V - Vcrit)/eps))

# =========================
# Algebraic current solver
# =========================
def solve_current(Vest, P, R0):
    def f(I):
        return R0*I**2 - Vest*I + P
    # initial guess: low-current branch
    I0 = P / max(Vest, 1e-3)
    sol = root(f, I0)
    if not sol.success:
        return None
    I = sol.x[0]
    delta = Vest**2 - 4*R0*P
    if delta <= 0:
        return None
    return I

# =========================
# Parameters
# =========================
dt = 0.5                      # s
Q = 4.0 * 3600                # Coulombs
R0 = 0.12
R1, C1 = 0.02, 2500
R2, C2 = 0.04, 800

# =========================
# Initial state
# =========================
soc = 1.0
vc1, vc2 = 0.0, 0.0
t = 0.0

T, SOC, Vt, Delta = [], [], [], []

# =========================
# Load profile
# =========================
P_base = 2.8  # gaming-like load

# =========================
# Simulation
# =========================
while soc > 0:

    Vest = ocv(soc) - vc1 - vc2
    lam = lambda_v(Vest)
    P = P_base * lam

    I = solve_current(Vest, P, R0)
    if I is None:
        print(f"⚠ Power collapse at t = {t/3600:.2f} h")
        break

    delta = Vest**2 - 4*R0*P

    # implicit-style update
    soc -= I * dt / Q
    vc1 += dt * (-vc1/(R1*C1) + I/C1)
    vc2 += dt * (-vc2/(R2*C2) + I/C2)

    V = ocv(soc) - vc1 - vc2 - I*R0

    T.append(t/3600)
    SOC.append(soc)
    Vt.append(V)
    Delta.append(delta)

    t += dt

# =========================
# Plot
# =========================
plt.figure(figsize=(10,8))

plt.subplot(3,1,1)
plt.plot(T, SOC)
plt.ylabel("SOC")
plt.grid(alpha=0.3)

plt.subplot(3,1,2)
plt.plot(T, Vt)
plt.ylabel("Voltage (V)")
plt.grid(alpha=0.3)

plt.subplot(3,1,3)
plt.plot(T, Delta)
plt.axhline(0, ls='--', c='r')
plt.ylabel("Δ")
plt.xlabel("Time (h)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Fig6.1_Numerical_Validation_of_DAE_Power_Constraint.png", dpi=300)
plt.show()
