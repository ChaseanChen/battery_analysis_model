import numpy as np
import matplotlib.pyplot as plt

# ====== 直接复用你已有的物理模型 ======
from battery_dae_simulation import get_ocv, get_resistance


def critical_soc(T_celsius, N_cyc, P_load, R_scale=1.0):
    soc_grid = np.linspace(0.99, 0.01, 2000)

    V = get_ocv(soc_grid)
    R = get_resistance(T_celsius, N_cyc) * R_scale
    Delta = V**2 - 4 * R * P_load

    sign_change = np.where(np.diff(np.sign(Delta)))[0]

    if len(sign_change) == 0:
        return np.nan

    i = sign_change[0]

    # ---- 线性插值 Δ(soc)=0 ----
    soc1, soc2 = soc_grid[i], soc_grid[i + 1]
    d1, d2 = Delta[i], Delta[i + 1]

    soc_crit = soc1 - d1 * (soc2 - soc1) / (d2 - d1)
    return soc_crit



def sensitivity_R0(T_celsius, N_cyc, P_load, perturb=0.01):
    """
    计算 TTE (SOC_crit) 对 R0 的无量纲灵敏度
    S = (ΔSOC / SOC) / (ΔR / R)
    """
    # ---- 基准 ----
    soc0 = critical_soc(T_celsius, N_cyc, P_load)
    if np.isnan(soc0):
        return np.nan

    soc0 = critical_soc(T_celsius, N_cyc, P_load, R_scale=1.0)
    soc1 = critical_soc(T_celsius, N_cyc, P_load, R_scale=1 + perturb)

    # ---- 相对灵敏度 ----
    S = ((soc1 - soc0) / soc0) / perturb
    return S


def sensitivity_map():
    """
    在功率可行域边界 (Δ=0) 处，
    计算灵敏度 S_R0 随 (T, N) 的变化
    """
    # ---- 网格设置 ----
    T_grid = np.linspace(-10, 45, 60)       # 温度
    N_grid = np.linspace(0, 800, 60)        # 老化循环数

    P_load = 8.0                            # 固定负载功率

    S = np.zeros((len(N_grid), len(T_grid)))

    for i, N in enumerate(N_grid):
        for j, T in enumerate(T_grid):
            S[i, j] = sensitivity_R0(T, N, P_load)

    # ---- 作图 ----
    plt.figure(figsize=(8, 5))
    # im = plt.contourf(
    #     T_grid,
    #     N_grid,
    #     np.abs(S),
    #     levels=30
    # )
    # plt.colorbar(im, label=r"$|S_{R_0}|$")
    plt.contourf(
    T_grid,
    N_grid,
    np.log10(np.abs(S) + 1e-3),
    levels=30
    )
    plt.colorbar(label=r"$\log_{10} |S_{R_0}|$")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Cycle Number")
    plt.title("Sensitivity of TTE to Internal Resistance\n(Evaluated at Power Feasibility Boundary)")
    plt.tight_layout()
    plt.savefig(
        "Fig5.3_Sensitivity_TTE_vs_R0.png",
        dpi=300
    )
    plt.show()


if __name__ == "__main__":
    sensitivity_map()
