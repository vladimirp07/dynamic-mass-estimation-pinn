# Dynamic Payload Estimation in Commercial Vehicles using PINNs

This repository implements and compares **Physics-Informed Neural Networks (PINNs)** for real-time payload (mass) estimation in commercial vehicles using telemetry data.

The project evaluates two primary data acquisition methodologies:
1. **Direct Mechanical Model (Suspension Sensor):** Reconstructs vertical displacement to estimate payload. For the Peugeot Partner van, it achieves a Gaussian error distribution ($\mu \approx 14.6 \text{ kg}$, $R^2 \approx 0.94$).
2. **Indirect Thermodynamic Model (OBD Engine Data):** Resolves longitudinal power balance equations based on fuel consumption telemetry. It serves as an alternative for vehicles without dedicated suspension displacement instrumentation.

While the modular codebase (`.py` files) retains the parameters and data pipelines for other vehicles (such as the **Freightliner Tractocamión** and the **RAM 1500**), the main orchestrator focuses on the **Peugeot Partner**, which represents the validated core methodology of this research.

---

## Repository Structure

The project is organized as a clean, modular python package suitable for research and production:

```text
├── data/                       # Raw and processed telemetry datasets
│   ├── 07 OBD RAM 29.09 jih.xlsx
│   ├── 09 OBD Peugeot 10.10 os.xlsx
│   ├── 09 OBD Peugeot Datos Limpios(Sheet1) - V2.csv
│   ├── 09 OBD Peugeot Datos Limpios(Sheet1).csv
│   ├── Metodo payload tracto.xlsx
│   └── Metodo payload tracto_Datos Limpios.xlsx
├── legacy/                     # Original Jupyter Notebooks containing development iterations
│   ├── NB1_Test1_OBD_Peugeot.ipynb
│   ├── NB1_Test1_OBD_Tracto.ipynb
│   └── NB1_Test1_Sensor_Peugeot.ipynb
├── src/                        # Core Python library modules
│   ├── models.py               # PyTorch neural network definitions (ThermodynamicPINN, SuspensionPINN)
│   ├── physics.py              # Physics loss formulations (longitudinal power balance, suspension force equations)
│   ├── data_utils.py           # Preprocessing pipelines, deflection conversions, and Tripwire filters
│   ├── train.py                # Training engines (window-based incremental and memory-inherited loops)
│   └── visualizations.py       # Evaluation reports and plot templates (KDE errors, temporal plots)
├── orchestrator.ipynb           # Main Jupyter orchestrator executing Peugeot Partner validations
└── README.md                   # Repository overview
```

---

## Physical Formulations

### 1. Mechanical Suspension Model
Models the rear axle of the Peugeot Partner furgoneta as a non-linear spring-damper-mass system. The vertical deflection of the suspension ($x$) is related to spring forces using quadratic stiffness coefficients:

$$M \frac{d^2x}{dt^2} + C \frac{dx}{dt} + (K_1 \cdot x + K_2 \cdot x^2) - (M - M_{base}) g = 0$$

*   **PINN Loss Function:** The model minimizes the mean squared error of the sensor displacement (data loss, scaled in mm²) and the residual of the suspension force equation (physics loss, scaled in kN²).

### 2. Thermodynamic Power Balance Model
Equilibrates available combustion power ($P_{disp}$) generated from fuel rate with the power required to overcome longitudinal resistance forces ($P_{req}$):

$$F_{req} = F_{drag} + F_{rolling} + F_{gravity} + F_{inertia}$$
$$F_{req} = \frac{1}{2} C_D A_f \rho_{air} V^2 + f_r M g \cos(\theta) + M g \sin(\theta) + M m_f \frac{dV}{dt}$$

$$P_{req} = \max(0, F_{req} \cdot V)$$
$$P_{disp} = \rho_{fuel} \cdot (\text{Fuel Flow} - \text{Idle Burn}) \cdot LHV_f \cdot \eta$$

---

## Vehicle Parameter Configurations

The codebase supports physical settings configured for multiple vehicle classes:

| Parameter / Constant | Van (Peugeot Partner) | Pick-up (RAM 1500 - Adaptable) | Heavy Truck (Freightliner Tracto) |
| :--- | :--- | :--- | :--- |
| **Base Mass (`M_BASE`)** | $1301 \text{ kg}$ (Tare + driver) | $\approx 2200 \text{ kg}$ (Estimated) | $9480 \text{ kg}$ (Tare + diesel + driver) |
| **Aerodynamic Area (`A_F`)**| 2.5 m² | 3.2 m² | 10.5 m² |
| **Drag Coeff. (`C_D`)** | $0.35$ | $0.38$ | $0.60$ |
| **Engine Efficiency ($\eta$)**| $12\% - 15\%$ (Gasoline) | $15\% - 18\%$ (Gasoline) | $40\%$ (Heavy Diesel) |
| **Lower Heating Value (`LHV`)**| $43 \text{ MJ/kg}$ | $43 \text{ MJ/kg}$ | $43 \text{ MJ/kg}$ |
| **Idle Fuel Rate (`Idle`)**| $0.0018 \text{ L/s}$ | $0.0022 \text{ L/s}$ | $0.0 \text{ L/s}$ (Drawn from diesel curve) |
| **Suspension Stiffness ($K_1$)**| $84705.20 \text{ N/m}$ | N/A | N/A |
| **Suspension Stiffness ($K_2$)**| 2903352.28 N/m²| N/A | N/A |

---

## Tripwire Logic

In thermodynamic mass estimation, transient anomalies (such as mechanical braking energy loss not registered in fuel consumption data, or clutching events) can corrupt the optimization gradients. To solve this, a sequence of logical gates (Tripwire Logic) filters out corrupt windows. A time window is discarded if:

1.  **Idling / Traffic:** Mean speed falls below $3.0 \text{ m/s}$.
2.  **Low Load Signal:** The payload is lower than the environmental noise envelope ($<70 \text{ kg}$ on lightweight vehicles).
3.  **Active Deceleration:** Braking occurs in over $40\%$ of the window (which would collapse PINN mass output to zero to balance the power equation).
4.  **Inertial Coefficient Spike:** Dynamic inertia coefficient `mf` rises above $30.0$.
5.  **Fuel Flow Outlier:** Instantaneous fuel rate exceeds sensor threshold ($>0.006 \text{ L/s}$ Peugeot).
6.  **Topographic Irregularity:** Slope variance $\text{std}(\theta) > 0.1 \text{ rad}$ (prevents gravitational oscillations from destabilizing parameters).

---

## Getting Started

### 1. Prerequisites
Install package dependencies in a Python environment:

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### 2. Execution
Run `orchestrator.ipynb` within your preferred notebook interface (Jupyter Lab, VS Code, etc.). 

The notebook imports the modular libraries, runs the Peugeot Partner mechanical suspension simulation ($R^2 \approx 0.94$) and the thermodynamic power simulation, and generates evaluation plots.
