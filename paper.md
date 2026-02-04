
# 考虑老化机制与系统反馈的智能手机锂离子电池 TTE 预测模型

**A Multi-Physics Continuous-Time Mechanistic Model for Smartphone Battery Discharge Prediction Considering Aging and System Feedback**

  

## 1. 引言 (Introduction)

  

背景：非线性耦合导致的系统失稳


现代智能手机的电源管理不仅受限于电化学储能，更受限于负载功率对端电压的非线性正反馈。在低温与老化场景下，电池内阻 $R_0$ 显著升高，使得功率平衡方程 $P=V⋅I$ 演变为一个典型的代数环问题。当电池无法满足负载的瞬时功率需求时，系统将跨越物理可行域的边界，导致瞬态电压崩溃（Voltage Collapse）。本文旨在构建一个 Index-1 的微分代数方程（DAE）系统，刻画这种从稳态运行到动力学失稳的临界演化过程。
  

传统的库伦计数法（Coulomb Counting）和开环等效电路模型作出的假设：

  

1. 负载功率恒定或外部给定；

2. 电池参数缓慢变化；

3. 电压-电流关系始终存在唯一的物理解；

  

这些假设在**高负载+老化+低温**等场景下会出现显著失真的情况，已无法应对这些非线性工况。

  

主要需要克服的问题：

  

1. **多物理场耦合**：电化学、热力学与电路特性的强耦合，以及与等效电路参数的高度依赖；

2. **老化效应 (Aging Effect)**：循环老化驱动的非线性退化，随着循环次数增加，内阻升高与容量衰减导致的性能退化，容量衰减与内阻增长对系统稳定性的影响远大于对 SOC 的影响；

  

3. **负载随机性与系统反馈**：负载不仅随机，还受操作系统的热节流（Throttling）机制控制，同时，操作系统的热节流与低压保护使负载功率成为电池状态的隐函数，形成代数环。

  

并结合我们建立一个包含 **SOH（健康状态）**、**电压滞后** 和 **闭环功率控制** 的连续时间微分-代数方程（DAE）模型，以精确预测剩余放电时间（TTE）及电压崩溃临界点的目标，我们决定在本文中共提出一个连续时间，多物理场耦合，含老化与系统反馈的微分-代数方程(DAE)模型，用于预测：

  

1. 剩余可用时间 Time-to-Empty, TTE;

  

2. 电压崩溃的临界边界；

  

3. 不同老化程度下的安全运动功率区间；

  

---


| 符号 (Symbol) | 定义 (Definition) | 单位 (Unit) |
| :--- | :--- | :--- |
| $SOC(t)$ | 电池荷电状态 (State of Charge) | 无量纲 (0~1) |
| $V_{C1}(t), V_{C2}(t)$ | 极化电容支路电压 (Polarization Voltages) | V |
| $H(t)$ | 滞后状态变量 (Hysteresis State) | 无量纲 (-1~1) |
| $T(t)$ | 电池核心温度 (Battery Core Temperature) | K |
| $I_{batt}(t)$ | 电池放电电流 (Discharge Current) | A |
| $V_{term}(t)$ | 电池端电压 (Terminal Voltage) | V |
| $V_{OCV}$ | 包含滞后的开路电压 (OCV with Hysteresis) | V |
| $V_{eq}$ | 电池平衡电势 (Equilibrium Potential) | V |
| $R_0, R_1, R_2$ | 欧姆/快极化/慢极化内阻 (Internal Resistances) | $\Omega$ |
| $C_1, C_2$ | 极化电容 (Polarization Capacitances) | F |
| $Q_{max}(N_{cyc})$ | 当前循环下的最大可用容量 (Actual Capacity) | Ah |
| $N_{cyc}$ | 电池等效循环次数 (Cycle Number) | 无量纲 |
| $P_{base}(t)$ | 系统基础功率需求 (Base Power Demand) | W |
| $P_{req}(t)$ | 经节流后的实际负载功率 (Actual Load Power) | W |
| $\lambda(V, T)$ | 系统功率反馈节流因子 (Throttling Coefficient) | 无量纲 (0~1) |
| $\eta_{pmic}$ | 电源转换效率 (Power Conversion Efficiency) | 无量纲 (0~1) |
| $m C_p$ | 电池等效热容 (Total Heat Capacity) | J/K |
| $hA$ | 综合散热系数 (Heat Transfer Coefficient) | W/K |
| $T_{amb}$ | 环境温度 (Ambient Temperature) | K |
| $E_a$ | 内阻活化能 (Activation Energy) | J/mol |
| $TTE$ | 剩余放电时间 (Time-to-Empty) | min |

  

## 2. 模型假设 (Model Assumptions)

![Fig2.1](./src/Fig2.1_NASA_Capacity_Degradation.png)

Figure X shows the capacity degradation trajectories of four lithium-ion cells from the NASA battery aging dataset.
Despite similar initial capacities, substantial dispersion in degradation rates is observed, motivating the introduction of cycle-dependent aging parameters and bounded uncertainty in the proposed model.

1. 集总参数与均匀性：将电池视作一个集总系统，假设电池内部温度、浓度，电化学状态在单体尺度上的分布均匀，该假设适用于手机单体电芯在中低倍率（≤2C）放电工况；

  

2. 双重老化机制：电池衰退由“日历老化”（忽略）和“循环老化”组成，主要表现为 SEI 膜增厚导致的内阻增加和活性锂损失导致的容量衰减。；

  

3. 个体差异正态分布：同批次电池的初始参数（$R_0, Q_{nom}$）服从正态分布 $N(\mu, \sigma^2)$，所以采用同批次电池在初始容量与内阻上的离散性通过正态分布扰动项建模，The stochastic perturbations are introduced solely to reflect manufacturing variability and are not treated as random variables for inference, but as bounded uncertainty terms constrained by empirical degradation envelopes.

**Separation of Scales and Quasi-static Approximation**: The timescale of capacity degradation and resistance growth (cycles/months) is several orders of magnitude larger than that of a single discharge event (minutes/hours). Consequently, aging-dependent parameters ($Q_{max}, R_{aging}$) are treated as **quasi-static constants** during the TTE prediction interval $[0, t_{end}]$. They are initialized as functions of $N_{cyc}$ at $t=0$ and remain invariant during the integration of the DAE system.
(**尺度分离与准静态近似**：电池老化过程（$N_{cyc}$）的时间尺度（天/月）远大于单次放电过程的时间尺度（分钟/小时）。因此，在单次 TTE 预测仿真中，老化相关参数（$Q_{max}, R_{aging}$）被视为**准静态常数（Quasi-static constants）**，其值仅在仿真初始时刻根据 $N_{cyc}$ 确定，在积分过程中不随 $t$ 演化。)
  

4. 分级关机逻辑假设：

- 软关机 (Soft Shutdown)：$V_{term} \le 3.4V$ 时系统触发省电模式（限制功率）；

- 硬关机 (Hard Cutoff)：$V_{term} \le 3.0V$ 或发生功率崩溃（Power Collapse）即代数约束上不存在实根时，强制断电；

In the proposed model, hard cutoff is triggered either by voltage protection or by the loss of feasibility of the algebraic power constraint.





---


## 3. 数学建模 (Mathematical Modeling)


![Fig3.0](./src/Fig3.0_Architecture_of_Coupled_DAE_Battery_System.png.png)

Figure 3.0. Architecture of the coupled DAE-based battery discharge model with aging, thermal dynamics, and closed-loop power feedback.


Electrical dynamics, thermal evolution, aging states, and system-level feedback are **bidirectionally coupled**, producing intrinsic algebraic loops between terminal voltage, current, and power demand. Consequently, the system cannot be represented by a pure ODE and must be formulated as a **DAE system**.

  

定义系统状态向量：

  

$\mathbf{x}(t) = [SOC(t), V_{C1}(t), V_{C2}(t), T(t), H(t)]^T$。

  

### 3.1 子模块 1: 考虑老化与个体差异的参数模型 (Aging & Variance)

  

在仿真开始前，根据电池的循环次数 $N_{cyc}$ 和个体差异因子 $\delta$，确定当前电池的物理参数。


Despite similar initial capacities, substantial dispersion in degradation rates is observed, motivating the introduction of stochastic aging parameters and cycle-dependent capacity fade in the proposed model.

  

|Battery|Cycles|Capacity Fade|
|:-----:|:----:|:-----------:|
| B0005 | 168  |   28.62%    |
| B0006 | 168  |   41.75%    |
| B0007 | 168  |   24.25%    |
| B0018 | 132  |   7.71%     |

  

These values are used to constrain the feasible range of aging-related parameters in the proposed model.

  

**A. 容量衰减 (Capacity Fade)**

遵循 SEI 膜生长的平方根法则：

$$ Q_{max}(N_{cyc}) = Q_{design} \cdot (1 + \delta_Q) \cdot \left( 1 - \alpha_{sei} \sqrt{N_{cyc}} \right) $$

*其中 $\delta_Q \sim N(0, 0.02)$ 为个体容量差异。*

The square-root law is applied at the cycle level and does not imply continuous-time differentiability of capacity with respect to time.

A lower bound is imposed on (Q_{max}) to prevent nonphysical negative capacity during long-term extrapolation.

  

Based on the NASA lithium-ion battery aging dataset (B0005, B0006, B0007, B0018), the observed capacity fade after 130–170 discharge cycles ranges from **24.25% to 41.75%**, indicating a non-negligible dispersion across cells.

  

Therefore, the parameter (\alpha_{sei}) is calibrated such that

$$

\alpha_{sei} \in [0.018, 0.032]

$$

ensuring consistency with empirical degradation trajectories.

  

**B. 内阻增长 (Resistance Growth)**

内阻随循环次数线性或指数增长（取决于化学体系，此处采用线性简化）：

$$ R_{aging}(N_{cyc}) = (1 + \beta_{res} N_{cyc}) \cdot (1 + \delta_R) $$

*其中 $\delta_R \sim N(0, 0.03)$ 为个体阻抗差异。*

  

Empirical studies on the NASA dataset indicate that, Within the first 150–200 cycles, experimental observations indicate that the effective DC internal resistance increases monotonically and can be approximated by a linear function.

Therefore, a linearized aging model is adopted to balance physical fidelity and numerical tractability.

  

---

  

### 3.2 子模块 2: 闭环负载与反馈控制 (Closed-loop Load Model)

  

为了模拟真实的手机系统，负载不再是恒定的，而是受电压和温度反馈控制的变量。

  

**A. 基础功率需求**

$$ P_{base}(t) = \frac{1}{\eta} \left( P_{disp}(t) + P_{cpu}(t) + P_{net}(t) \right) $$



  

**B. 系统反馈因子 (System Throttling)**

引入调节因子 $\lambda(t) \in [0, 1]$，模拟电源管理逻辑：

$$ \lambda(t) = \min \left( \underbrace{\frac{1}{1+e^{(T - 45)/2}}}_{\text{Thermal Throttling}}, \quad \underbrace{\tanh(k_v (V_{term} - 3.0))}_{\text{Low Voltage Throttling}} \right) $$

$(T - T_{crit}), \quad T_{crit}=45^\circ\mathrm{C}
$，Temperature is expressed in degrees Celsius in the throttling logic for consistency with system-level thermal limits.

可将其描述为一个软切换函数 Soft-switching function：

For analytical convenience, the throttling logic may alternatively be represented by a smooth soft-switching approximation with equivalent saturation behavior


![Fig3.2](./src/Fig3.2_Voltage_Based_Power_Throttling.png)


$$ \lambda(V) = \frac{1}{2} \left[ 1 + \tanh\left( \frac{V - V_{crit}}{\epsilon} \right) \right] $$
  

The throttling function is not required to be smooth everywhere, as it represents discrete system-level power management logic rather than intrinsic electrochemical dynamics.

  

**C. 实际负载**

实际负载功率 $P_{req}$ 是电压与温度的隐函数：


$$ P_{req}(t) = P_{base}(t) \cdot \lambda(t) $$
$$ P_{req} = P_{base} \cdot \lambda(V, T) $$
  

---

  

### 3.3 子模块 3: 电化学动态 (Electrochemical Dynamics)

  

**A. 容量归一化的 SOC 动态方程**


$$
\frac{dSOC}{dt} = -\frac{I_{batt}(t)}{Q_{max}},  
\qquad Q_{max} \text{ expressed in Coulombs (As)}
$$

  

**B. 包含滞后的 OCV 模型 (OCV with Hysteresis)**

引入滞后状态变量 $H(t)$ 以修正充放电电压差：

$$ V_{OCV}(SOC, H) = V_{eq}(SOC) + M(SOC) \cdot H(t) $$

The OCV curve is treated as an empirical monotonic function calibrated from typical smartphone Li-ion cells, rather than a chemistry-specific equilibrium potential.

滞后项的动态方程：

$$ \frac{dH}{dt} = -\kappa |I_{batt}| (H - \text{sgn}(I_{batt})) $$

*当持续放电时，$H \to -1$，表现为电压低于平衡电势 $V_{eq}$。*

In this work, only discharge scenarios are considered; therefore, the hysteresis state converges to (H \to -1), and the sign function reduces to a constant.


**C. 极化电压 (Dual-RC)**

$$ \frac{dV_{Ci}}{dt} = -\frac{1}{R_i C_i} V_{Ci} + \frac{1}{C_i} I_{batt}, \quad i=1,2 $$

  

---

  

### 3.4 子模块 4: 热力学耦合 (Thermodynamics)


![Fig3.4](./src/Fig3.4_R0_Temperature_Cycle_Surface.png)


这标志着系统进入了‘内阻占导（Resistance-Dominated）’态，此时电池更像一个发热器而非电源

**A. 阿伦尼乌斯-老化耦合内阻**

这是模型的核心非线性源：

$$ R_0(T, N_{cyc}) = R_{base} \cdot R_{aging}(N_{cyc}) \cdot \exp \left[ \frac{E_a}{R_g} \left( \frac{1}{T} - \frac{1}{T_{ref}} \right) \right] $$


**B. 热平衡方程**

$$
m C_p \frac{dT}{dt} = I_{batt}^2 R_0 - hA(T-T_{amb})
$$

  
Thermal parameters are selected to reproduce qualitative trends rather than calibrated absolute temperature profiles.


Only the ohmic resistance (R_0) is considered as the dominant heat generation source, while polarization losses are neglected for simplicity.


---

  

### 3.5 功率平衡约束与代数环分析 (System Coupling)

在传统的 $ODE$ 模型中，电流 $I$ 通常会作为外生输入变量，然而，再考虑到系统级的反馈的闭环模型中，电流时由瞬时功率平衡决定的隐函数；


At each time instant, the battery current is implicitly determined by enforcing instantaneous power balance, resulting in an algebraic constraint.

系统状态向量的演化受限于功率平衡流形（Power Balance Manifold），构成微分-代数方程组 (DAE)。

定义有效电压：

$$V_{est} = V_{OCV}(SOC, H) - V_{C1} - V_{C2}$$

端电压 $V_{term}$ 与电流 $I$ 的代数约束关系如下：

$$ V_{term} = V_{est} - I \cdot R_0(T, N_{cyc}) $$

$$ V_{term} = V_{OCV}(SOC, H) - V_{C1} - V_{C2} - I_{batt} R_0(T, N_{cyc}) $$

结合负载功率需求 $P_{req}$，功率约束方程（代数环）即功率平衡方程 $f(I, \mathbf{x}) = 0$，可得到：

$$ \boxed{ R_0(T, N_{cyc}) \cdot I_{batt}^2 - [V_{OCV}(SOC, H) - V_{C1} - V_{C2}] \cdot I_{batt} + P_{req}(t) = 0 } $$

$$ R_0(T, N_{cyc}) \cdot I^2 - V_{est}(SOC, V_{Ci}) \cdot I + P_{req}(V, T) = 0 $$

这是模型的动力学核心。定义有效电压 $Vest=VOCV−∑VCi$ ，电流 $I$ 由瞬时功率平衡决定：

$$ \boxed{ f(I) = R_0 I^2 - V_{est} I + P_{req} = 0 } $$

在传统的ODE模型中共，电流 $I_{batt}$ 通常是输入变量，但是在本模型中，由于系统的反馈和功率守恒的引入，电流演变为状态变量的隐函数；

公式实际上是一个关于 $I_{batt}$ 的非线性代数方程：

$$ f(I_{batt}) = R_0 I_{batt}^2 - V_{est} I_{batt} + P_{req} = 0 $$

其中 $V_{est} = V_{OCV} - V_{C1} - V_{C2}$

方程有解：

$$ I_{batt} = \frac{V_{est} \pm \sqrt{V_{est}^2 - 4 R_0 P_{req}}}{2 R_0} $$

从动力学角度来看，电流 $I$ 是由代数方程定义的隐函数，所以该方程的解的存在性取决于判别式：

$$\Delta = V_{est}^2 - 4 R_0 P_{req}$$

从数学的角度来观察，方程实际上存在两个实根，但是考虑到物理层面，我们必须选择带有负号的分支，即那个较小的根：

1. 负号分支$I^* = \frac{V_{est} - \sqrt{\Delta}}{2 R_0}$对应的是低电流，高电压的工作点，耗散率低，是电池的稳态运行区；
2. 非负号分支$I = \frac{V_{est} + \sqrt{\Delta}}{2 R_0}$对应在非常高电流情况下的低电压状态，在这种状态下，电池内阻所消耗的功率很可能会超越负载功率，这会导致正反馈出现热失控，在这种情况下，往往会导致瞬时电压坍塌 Voltage Collapse;

当 $\Delta \to 0$ 时，系统抵达物理可行域的边缘（Discriminant Manifold），此时 the Jacobian with respect to the algebraic variable $\partial f / \partial I$ becomes singular 在数值模拟中，这表现为系统刚性（Stiffness）的无穷大，物理上则对应于瞬间关机。

![fig4.1](./src/Fig4.1_DAE_SOC_and_Discriminant_OpenLoop.png)

Note that $Δ$ remains positive at $SOC≈0$ for this specific $Pbase$ , implying the cut-off is energy-limited rather than stability-limited.
注意在此功率下，$Δ$ 在 SOC 归零时仍为正，说明关机是受能量限制而非稳定性限制

**$DAE$指数分析：**

$Jacobian$ 矩阵为：
$$\frac{\partial f}{\partial I} = 2 R_0 I - V_{est}$$

由于 $\frac{\partial f}{\partial I_{batt}} = 2 R_0 I_{batt} - V_{est}$，只要判别式 $\Delta > 0$，则 $\left.\frac{\partial f}{\partial I}\right|_{I=I^{*}}=2 R_{0} I^{*}-V_{e s t}=-\sqrt{\Delta}$，这就意味着该系统依旧属于 $Index-1 DAE$ ；

这保证了我们可以使用常规的隐式求解器（如 $BDF$ 法或 $Radau$ 方法）进行稳定积分；

然而，当系统逼近临界点（$\Delta \to 0$）时，系统抵达物理可行域的边缘（Discriminant Manifold），此时 $Jacobian$ 矩阵 $\partial f / \partial I$ 趋于奇异（$\frac{\partial f}{\partial I} \to 0$）。在数值模拟中，这表现为系统上的数值刚性（Stiffness）的无穷大 $^1$；

The critical point of voltage collapse is analyzed as a **saddle-node bifurcation**, following the theoretical framework established by **Dobson and Chiang [5]**.

从动力学系统角度看，由于两个解分支在 $\Delta = 0$ 处相遇并消失，这构成了典型的鞍结分岔（Saddle-node Bifurcation）。这就是导致手机“瞬间关机”的numerical manifestation of the underlying saddle-node bifurcation：当负载功率或内阻的乘积超过电压平方的四分之一时，物理层面不再存在能够维持功率平衡的实数电流解，系统被迫从动力学稳态直接跳变至断电状态。

---

  

## 4. 稳定性分析与相平面 (Stability & Phase Plane Analysis)

为了量化“瞬间死机”的边界，我们不应仅将其视为一个数值报错，而应将其视为动力学系统中的鞍节点分叉 (Saddle-Node Bifurcation)。


![Fig4.2](./src/Fig4.2_Discriminant_Boundary_vs_SOC.png)

**A. 判别式流形 (Discriminant Manifold)**

通过数值仿真，我们可以观察到在开环恒功率放电过程中，判别式 $Δ$ 随着 SOC 的消耗持续逼近临界值 0，这验证了判别式流形作为系统动力学可行域边界的物理意义。

![fig4.3](./src/Fig4.3_Voltage_Collapse_OpenLoop.png)

The simulation shows the rapid terminal voltage decline under a high-power demand ( $Pbase=8W$ ). The absence of feedback leads the system directly toward the singular manifold $Δ=0$ .
仿真展示了高功率需求下端电压的快速跌落。由于缺乏反馈，系统直接走向奇异流形 $Δ=0$

对于二次方程，物理可解的条件是判别式 $\Delta \ge 0$。定义系统状态空间中的**崩溃边界**：

$$ \Delta(SOC, T, N_{cyc}) = [V_{est}(SOC) ]^2 - 4 \cdot R_0(T, N_{cyc}) \cdot P_{req} $$

*其中 $V_{est} = V_{OCV} - V_{pol}$ 为除去欧姆压降后的有效电压。*


**B. 相平面可视化 (Phase Plane Visualization)**

以 $SOC$ 为横轴，负载功率 $P$ 为纵轴，绘制不同老化程度和温度下的**安全运行区 (Safe Operating Area, SOA)**。

为了防止系统超越 4.B 中定义的崩溃边界 $Pcrit$ ，引入的闭环节流机制,能够动态调整负载功率，使得判别式 $Δ$ 始终保持在正值区间，从而避开了鞍节点分岔点。

![Fig4.4](./src/Fig4.4_Voltage_and_Discriminant_Controlled.png)

The feedback controller acts as a singular perturbation buffer, preventing the state trajectory from hitting the singular manifold where $∂f/∂I=0$ .
反馈控制器起到了奇异扰动缓冲器的作用，防止状态轨迹触碰 $∂f/∂I=0$ 的奇异流形

The feedback mechanism effectively regularizes the power demand as the voltage nears the threshold. Note that the discriminant $Δ$ is stabilized above zero, effectively preventing the saddle-node bifurcation and extending the operational duration (Case C).
当电压接近阈值时，反馈机制有效地调节了功率需求。注意判别式 $Δ$ 被稳定在零以上，有效地预防了鞍结分岔并延长了运行时间。

* **曲线方程**：$P_{crit}(SOC) = \frac{V_{est}(SOC)^2}{4 R_0(T)}$


In phase-plane analysis, polarization voltages are neglected to highlight the dominant effect of ohmic resistance and OCV.


* **分析**：

* 曲线 **下方** 为稳定运行区。

* 曲线 **上方** 为电压崩溃区（无法维持功率，电压瞬间坍塌）。

* **老化影响**：随着 $N_{cyc}$ 增加，$R_0$ 变大，曲线整体下移，SOA 面积急剧收缩。



The curves represent the discriminant boundary beyond which the algebraic power constraint admits no real solution, leading to voltage collapse.

As aging progresses or temperature decreases, the safe operating area (SOA) shrinks significantly, explaining the observed sudden shutdown phenomena under high-load applications such as mobile gaming.

  

---

  

## 5. 仿真流程与结果定义 (Simulation Procedure)


![Fig5.1](./src/Fig5.1_Power_Feasibility_Boundary_P8W.png)

### 5.1 数值求解策略

由于系统在截止点附近具有强刚性（Stiffness），不能简单使用欧拉法。

  

1. **求解器**：Python `scipy.integrate.solve_ivp` (采用 `BDF` 或 `Radau` 方法);

  

2. **每步迭代逻辑**：

  

* 根据当前状态 $(T, SOC, \dots)$ 更新 $R_0, V_{OCV}$;

* 计算反馈因子 $\lambda$ 得到 $P_{req}$;

* 求解代数约束方程得到 $I_{batt}$（取较小正根）;

  

* 若 $\Delta < 0$，判定为 **Power Collapse**，仿真终止;

  

* 计算状态导数 $d\mathbf{x}/dt$;

  

Although the system contains algebraic loops, the index of the DAE remains low due to the explicit quadratic form of the power constraint, enabling stable integration via implicit solvers.



### 5.2 预测指标输出

1. **TTE (Time-to-Empty)**：$\inf \{ t : SOC=0 \lor V_{term} \le 3.0V \lor \Delta < 0 \}$。



Once the terminal voltage approaches the soft shutdown threshold, the feedback controller gradually reduces the power demand, preventing immediate voltage collapse and extending the remaining operational time.

This behavior cannot be captured by traditional open-loop discharge models.



1. **关机类型判定**：

* 能量耗尽：$SOC \approx 0$ 且 $V_{term} > 3.0V$。

* 低压保护：$V_{term} \le 3.0V$ 且 $\Delta > 0$（常见于老化电池）。

* 瞬间死机：$\Delta < 0$（常见于低温+重负载）。



## 5.3 关键参数灵敏度分析 (Key Parameter Sensitivity Analysis)

为了定量评估系统在不同工况下的鲁棒性，本节分析了剩余放电时间（TTE）对内阻 $R_0$ 扰动的无量纲灵敏度；

由于 TTE 的终点由功率判别式 $\Delta = 0$ 确定的临界状态 $SOC_{crit}$ 决定，我们决定定义 $SOC_{crit}$ 对 $R_0$ 的相对灵敏度系数 $S_{R_0}$ 为：

$$ S_{R_0} = \frac{\partial SOC_{crit} / SOC_{crit}}{\partial R_0 / R_0} = \frac{R_0}{SOC_{crit}} \cdot \frac{d SOC_{crit}}{d R_0} $$

### 5.3.1 隐函数求导

根据代数约束

$$f(SOC, R_0) = V_{OCV}(SOC)^2 - 4 R_0 P_{req} = 0$$

我们可以利用隐函数求导得到：

$$ \frac{d SOC_{crit}}{d R_0} = - \frac{\partial f / \partial R_0}{\partial f / \partial SOC} = \frac{4 P_{req}}{2 V_{OCV}(SOC) \cdot \frac{d V_{OCV}}{d SOC}} $$

分析该式可知，灵敏度的大小高度依赖于 $V_{OCV}$ 曲线在临界点处的斜率以及当前电压水平。在 $SOC$ 较低的区域，$V_{OCV}$ 快速下降（即 $d V_{OCV} / d SOC$ 增大），导致 $d SOC_{crit} / d R_0$ 呈现非线性增长。

### 5.3.2 灵敏度热图分析

通过对温度 $T \in [-10, 45]^\circ\text{C}$ 和老化循环 $N_{cyc} \in [0, 800]$ 构成的参数空间进行分析，可计算出 $S_{R_0}$ 的分布；

![Fig5.3](./src/Fig5.3_Sensitivity_TTE_vs_R0.png.png)

为了覆盖多个数量级的变化，我们在图中采用了对数标尺 $\log_{10} |S_{R_0}|$ 进行可视化；

1.  **线性平稳区 (Low Sensitivity Region)**：
    在常温（$>25^\circ\text{C}$）且电池较新（$N_{cyc} < 200$）时，内阻 $R_0$ 较小，系统远离鞍结分岔点。此时 $|S_{R_0}|$ 维持在低位，TTE 的误差主要来源于容量 $Q_{max}$ 的估算，BMS 表现出较强的鲁棒性。

2.  **动力学敏感区 (High Sensitivity Region)**：
    随着温度降低或老化程度加深，$R_0$ 指数级增长使得系统压向物理可行域的边界。在低温老化区域（Fig3中左上角），灵敏度系数发生了数量级跳跃（$\log_{10} |S_{R_0}| > 1$）。

这解释了为什么传统 Coulomb Counting（库伦计数）在老化电池上频繁失效，也证明了本 DAE 模型在捕捉功率边界动态行为方面的必要性

在实际工程中，Fig3的分析建议 BMS 应当在低温或老化末期自动提升内阻参数的更新频率，以补偿这种高阶灵敏度带来的预测风险。


---


## 6. 结论 (Conclusion)

![Fig6.1](./src/Fig6.1_Numerical_Validation_of_DAE_Power_Constraint.png)

Figure 6.1 numerically validates that the loss of solvability of the algebraic power balance equation coincides with the discriminant condition Δ = 0, confirming the theoretical feasibility boundary.


本模型通过引入循环老化因子、滞后电压以及系统级反馈控制，弥补了传统库伦计数法的缺陷。

1. **老化预测**：量化了内阻增长对 TTE 的非线性缩减作用（例如：容量衰减 20%，但在重负载下 TTE 可能缩减 50%）。

2. **边界可视化**：相平面分析清晰地展示了“低温关机”的物理边界。

3. **实用价值**：该模型可部署于 BMS 中，提供比单纯 SOC 更具指导意义的“最大可用功率”预测。

### Data and Code Availability
The empirical degradation models in this study were calibrated using the publicly available **NASA PCoE Battery Dataset [1]** and **CALCE Battery Data [2]**. The system-level power throttling logic and thermal management parameters were modeled with reference to the **Android Open Source Project (AOSP) [8]**. The simulation framework and numerical implementation developed in this work referenced the architecture of the open-source project available at **[10]**.

---

### References

[1] B. Saha and K. Goebel, "Battery Data Set," *NASA Ames Prognostics Data Repository*, NASA Ames Research Center, Moffett Field, CA, 2007. [Online]. Available: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

[2] M. Pecht, "CALCE Battery Data," *Center for Advanced Life Cycle Engineering (CALCE)*, University of Maryland, 2023. [Online]. Available: https://calce.umd.edu/battery-data

[3] National Battery Safety Innovation Center, "Battery Operational Data Series," *NBSDC Data Repository*, 2022. [Online]. Available: https://nbsdc.cn/general/dataDetail?id=630c856999f1de3bca1e63c2

[4] R. Garg, "Mobile Battery Dataset with Time-Series Voltage Profiles," *Kaggle Repository*, 2023. [Online]. Available: https://www.kaggle.com/datasets/rahulgarg28/mobile-battery-with-time

[5] I. Dobson and H.-D. Chiang, "Towards a theory of voltage collapse in electric power systems," *Systems & Control Letters*, vol. 13, no. 3, pp. 253-262, 1989.

[6] J. F. Manwell and J. G. McGowan, "Lead acid battery storage model for hybrid energy systems," *Solar Energy*, vol. 50, no. 5, pp. 399-405, 1993. *(Note: Kinetic Battery Model foundation)*.

[7] L. Zhang et al., "Accurate online power estimation and automatic battery behavior based power model generation for smartphones," in *Proc. IEEE/ACM/IFIP Int. Conf. Hardware/Software Codesign and System Synthesis (CODES+ISSS)*, 2010.

[8] Google, "Android Power Management and Thermal Mitigation Strategy," *Android Open Source Project (AOSP) Documentation*, 2024. [Online]. Available: https://source.android.com/docs/core/power

[9] GitCode Tech Blog, "Analysis of Android Battery Historian Data Structures," 2023. [Online]. Available: https://blog.gitcode.com/f5f7ea29887236ee33a94c651a75c920.html

[10] R. Tan, "BatteryLife: Open Source Battery Aging Analysis Tools," *GitHub Repository*, 2024. [Online]. Available: https://github.com/Ruifeng-Tan/BatteryLife
