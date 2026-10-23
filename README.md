# PINN-Vanila

Physics-informed neural network for steady two-dimensional incompressible flow through a smooth planar bend. The package implements the supplied parameter family: channel width 0.5 m, centerline radius 1 m, 2 m inlet and outlet segments, bend angles from 15° to 90°, and inlet `Vmax` from 1 to 20 m/s (`Re=500..10000`).

The default model solves steady incompressible **RANS with the Wilcox-1988 k-omega closure**, predicting mean velocity, static pressure, turbulent kinetic energy k and specific dissipation omega. It uses a float64 geometry-conditioned residual MLP with FiLM and Fourier features. Inlet/wall constraints are exact by construction; training includes mean-flow and k/omega transport residuals, soft outlet gradients, mass-flux control, a Reynolds/angle curriculum and residual adaptive refinement. See [RANS equations, boundaries and validation](docs/rans.md) for the exact closure and finite wall-omega regularization.

The previous laminar solver remains available through `configs/pinn_laminar_config.yaml`. Old three-output models remain laminar and require new training to obtain a RANS checkpoint.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -m pip install --no-cache-dir -e .
```

Validate the supplied geometry, fluid properties, mappings, and hard boundary conditions:

```bash
.venv/bin/pinn-vanila validate-config --config configs/pinn_vanila_config.yaml
```

## Training

The complete configured run is intentionally expensive: four curriculum stages with as many as 50,000 Adam steps per stage, second derivatives in float64, RAR candidate pools, and three independent seeds required by the experimental plan. Start one seed with:

```bash
.venv/bin/pinn-vanila-train \
  --config configs/pinn_vanila_config.yaml \
  --output trained/rans-komega/seed-42
```

Repeat with `--seed 137` and `--seed 2026` and distinct output directories for the other required independent runs.

Use small overrides only for an execution smoke test:

```bash
.venv/bin/pinn-vanila-train \
  --config configs/pinn_vanila_config.yaml \
  --output /tmp/pinn-smoke --quick
```

The module is also directly runnable with `python -m pinn_vanila.train --output /tmp/pinn-smoke --quick --device cuda`.
In VS Code, open `src/pinn_vanila/train.py` and click **Run Python File** to start the full configured GPU RANS training; its default checkpoint directory is `trained/rans-komega/seed-42`. CUDA is mandatory by default, so a machine without an attached NVIDIA GPU fails clearly instead of silently using the CPU. Use `--device cpu` only for a short debugging run.

The model directory records the run as it proceeds:

- `training.log` contains timestamped console logs with total, continuity, momentum, k/omega transport, outlet-boundary, and mass-flux losses.
- `metrics.csv` contains loss values and loss weights at every configured logging step; `metrics.json` adds field diagnostics and structural checks.
- `training_losses.png` plots mean-flow, boundary, flux, k/omega, and total loss histories.
- `fields/fields_step_*.png` contains velocity components, speed, inlet-gauged relative pressure, k, omega, eddy viscosity and velocity vectors every 200 steps, plus the final step.

A saved directory contains `model.pt` plus `metadata.json`, including the physical domain and validation status. A trained model is not labeled CFD-validated until it has been compared with independent, boundary-condition-matched reference data.

## Python API

```python
import numpy as np
from pinn_vanila import FlowCase, PINN

case = FlowCase.from_yaml(
    "configs/geometry_parametric.yaml",
    "configs/fluid_selected.yaml",
)
model = PINN.load("trained/rans-komega/seed-42")
points_xy_m = np.array([[0.5, 0.0], [2.0, 0.0]])
result = model.predict(
    geometry=case.geometry,
    points_xy=points_xy_m,
    inlet_vmax_m_s=12.0,
    inlet_pressure_pa=180000.0,
)
print(result.u_m_s, result.v_m_s, result.p_pa)
print(result.k_m2_s2, result.omega_s_inv, result.nu_t_m2_s)
```

The inlet pressure is the mean static pressure at the inlet section and fixes only the additive pressure gauge. The package computes

`p = p_in + rho * Vmax**2 * (pi - mean_inlet(pi))`.

Queries outside the channel or trained speed/angle range fail explicitly. `extrapolation_override=True` permits an intentional speed or angle extrapolation; different fixed channel dimensions always require retraining.

## CLI inference and export

```bash
.venv/bin/pinn-vanila predict \
  --model trained/rans-komega/seed-42 \
  --geometry configs/geometry_parametric.yaml \
  --fluid configs/fluid_selected.yaml \
  --points points.csv --vmax 12 --pressure 180000 \
  --output prediction.csv
```

CSV export is built in. HDF5 and VTU/VTK export are available when `h5py` and `meshio` are installed, respectively.
Install both optional exporters with `.venv/bin/python -m pip install -e '.[io]'`.

Compare a checkpoint with boundary-condition-matched CFD data (CSV columns `x_m,y_m,u_m_s,v_m_s,p_pa`):

```bash
.venv/bin/pinn-vanila evaluate \
  --model trained/rans-komega/seed-42 \
  --geometry configs/geometry_parametric.yaml \
  --reference reference.csv --vmax 12 --pressure 180000 \
  --output validation-report.json
```

The report aligns the pressure gauge, keeps velocity and pressure-drop errors separate, and adds mass balance, wall slip, outlet backflow, outlet-gradient, and PDE-residual diagnostics.

## Validation boundary

The combined zero normal gradients for outlet pressure and velocity are an engineering closure, not a universal well-posed open boundary. Poiseuille flow has `dp/dx = -8 μ Vmax/H²`; its exact benchmark therefore uses that compatible nonzero outlet gradient. High-Re stationary solutions and the provisional 5% velocity/pressure-drop targets must be demonstrated against independent CFD for each `(Re, angle)` region rather than inferred from training loss.
