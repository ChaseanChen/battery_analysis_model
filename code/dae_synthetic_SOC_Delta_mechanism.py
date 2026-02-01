import numpy as np
import matplotlib.pyplot as plt

def ocv(soc):
    return 3.0 + 1.15*soc - 0.18*np.exp(-10*soc)

def solve_current(Vest, P, R0):
    Delta = Vest**2 - 4*R0*P
    if Delta <= 0:
        return None, Delta
    I = (Vest - np.sqrt(Delta)) / (2*R0)
    return I, Delta

# ======================================================
# Pure mechanism simulation (Case A only)
# ======================================================
dt = 0.5
Q = 4.0 * 3600
R0 = 0.12
R1, C1 = 0.02, 2500
R2, C2 = 0.04, 800

soc = 1.0
vc1, vc2 = 0.0, 0.0
t = 0.0
P = 8.0

T, SOC, Delta_hist = [], [], []

while soc > 0:
    Vest = ocv(soc) - vc1 - vc2
    I, Delta = solve_current(Vest, P, R0)
    if I is None:
        break

    soc -= I * dt / Q
    vc1 += dt * (-vc1/(R1*C1) + I/C1)
    vc2 += dt * (-vc2/(R2*C2) + I/C2)

    T.append(t/3600)
    SOC.append(soc)
    Delta_hist.append(Delta)

    t += dt

# ======================================================
# Plot (Horizontal Layout: SOC and Delta side-by-side)
# ======================================================
# 将画布改为 1行2列，宽度设为12，高度设为5
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左侧子图: SOC (Figure 4a)
axes[0].plot(T, SOC, lw=2, color='tab:blue')
axes[0].set_ylabel("SOC", fontsize=10)
axes[0].set_xlabel("Time (h)", fontsize=10) # 左右布局时，左图也需要横坐标标签
axes[0].set_title("Simulated SOC under open-loop constant-power discharge", fontsize=11)
axes[0].grid(alpha=0.3)

# 右侧子图: Delta (Figure 4b)
axes[1].plot(T, Delta_hist, lw=2, color='tab:orange')
axes[1].axhline(0, ls="--", c="r", label="Bifurcation Threshold")
axes[1].set_ylabel("$\Delta = V^2 - 4R_0P$", fontsize=10)
axes[1].set_xlabel("Time (h)", fontsize=10) # 右图横坐标标签
axes[1].set_title("Evolution of discriminant $\Delta$ (saddle-node bifurcation)", fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout() # 自动优化子图间距
plt.savefig("Fig4.1_DAE_SOC_and_Discriminant_OpenLoop.png", dpi=300)
plt.show()