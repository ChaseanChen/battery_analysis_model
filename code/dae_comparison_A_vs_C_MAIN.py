"""
SOC asymptotically approaches a residual value under throttling.

对图需要作解释：
In Case C, SOC asymptotically approaches a residual value and the simulation is truncated at TmaxT_{max}Tmax​ to avoid excessively long low-power tails.

"""


import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Core battery model
# ======================================================
def ocv(soc):
    return 3.0 + 1.15*soc - 0.18*np.exp(-10*soc)

def lambda_v(V, enable=True, Vcrit=3.4, eps=0.05):
    if not enable:
        return 1.0
    return 0.5 * (1 + np.tanh((V - Vcrit)/eps))

def solve_current(Vest, P, R0):
    Delta = Vest**2 - 4*R0*P
    if Delta <= 0:
        return 0.0, Delta
    I = (Vest - np.sqrt(Delta)) / (2*R0)
    return I, Delta

# ======================================================
# Simulation runner
# ======================================================
def run_simulation(enable_control):
    dt = 0.5
    Q = 4.0 * 3600
    R0 = 0.12
    R1, C1 = 0.02, 2500
    R2, C2 = 0.04, 800

    soc = 1.0
    vc1, vc2 = 0.0, 0.0
    t = 0.0

    T, Vt, Delta_hist = [], [], []

    P_base = 8.0
    T_max = 50.0
    I_min = 1e-4
    
    while soc > 0 and t/3600 < T_max:
        Vest = ocv(soc) - vc1 - vc2
        lam = lambda_v(Vest, enable=enable_control)
        P = P_base * lam

        I, Delta = solve_current(Vest, P, R0)
        if I < I_min:
            break

        soc -= I * dt / Q
        vc1 += dt * (-vc1/(R1*C1) + I/C1)
        vc2 += dt * (-vc2/(R2*C2) + I/C2)

        V = ocv(soc) - vc1 - vc2 - I*R0

        T.append(t/3600)
        Vt.append(V)
        Delta_hist.append(Delta)

        t += dt

    return np.array(T), np.array(Vt), np.array(Delta_hist)

# ======================================================
# Run simulations
# ======================================================
TA, VA, DA = run_simulation(enable_control=False)
TC, VC, DC = run_simulation(enable_control=True)



# ======================================================
# Plot
# ======================================================


plt.figure(figsize=(6, 4))
plt.plot(TA, VA, lw=2, color='tab:blue')
plt.title("Case A: Open-loop Voltage", fontsize=12)
plt.ylabel("Voltage (V)", fontsize=10)
plt.xlabel("Time (h)", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Fig4.3_Voltage_Collapse_OpenLoop.png", dpi=300)
plt.show()

# 第二张图：Case C 的两个子图 左右排列 (Figure 6 & 7)
# figsize 改为 (12, 4.5) 以适应横向布局
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5)) 

# 左侧子图：Case C Voltage (Figure 6)
axes[0].plot(TC, VC, lw=2, color='tab:green')
axes[0].set_title("Case C: Controlled Voltage", fontsize=12)
axes[0].set_ylabel("Voltage (V)", fontsize=10)
axes[0].set_xlabel("Time (h)", fontsize=10)
axes[0].grid(alpha=0.3)

# 右侧子图：Case C Discriminant (Figure 7)
axes[1].plot(TC, DC, lw=2, color='tab:red')
axes[1].axhline(0, ls="--", c="black", alpha=0.7, label="Bifurcation Boundary")
axes[1].set_title("Case C: Discriminant $\Delta$", fontsize=12)
axes[1].set_ylabel("$\Delta = V^2 - 4R_0P$", fontsize=10)
axes[1].set_xlabel("Time (h)", fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout() # 自动调整子图间距，防止标签重叠
plt.savefig("Fig4.4_Voltage_and_Discriminant_Controlled.png", dpi=300)
plt.show()