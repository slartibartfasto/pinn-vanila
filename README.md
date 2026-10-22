# PINN-Vanila

Physics-informed neural network for steady two-dimensional incompressible flow through a smooth planar bend. The package implements the supplied parameter family: channel width 0.5 m, centerline radius 1 m, 2 m inlet and outlet segments, bend angles from 15° to 90°, and inlet `Vmax` from 1 to 20 m/s (`Re=500..10000`).

The model is a float64 geometry-conditioned residual MLP with FiLM conditioning and two-scale Fourier features. It predicts dimensionless velocity and pressure. The inlet parabola and wall no-slip condition are exact by construction. Training uses physical-coordinate autograd for the strong Navier–Stokes residuals, soft outlet gradients, mass-flux control, a Reynolds/angle curriculum, gradient-norm loss balancing, and residual adaptive refinement.

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
.venv/bin/pinn-vanila train \
  --config configs/pinn_vanila_config.yaml \
  --output trained/seed-42
```

Repeat with `--seed 137` and `--seed 2026` and distinct output directories for the other required independent runs.

Use small overrides only for an execution smoke test:

```bash
.venv/bin/pinn-vanila train \
  --config configs/pinn_vanila_config.yaml \
  --output /tmp/pinn-smoke \
  --max-steps 1 --interior-batch 8 --boundary-batch 8 --no-rar --skip-polishing
```

A saved directory contains `model.pt` plus `metadata.json`, including the physical domain and validation status. A trained model is not labeled CFD-validated until it has been compared with independent, boundary-condition-matched reference data.

## Python API

```python
import numpy as np
from pinn_vanila import FlowCase, PINN

case = FlowCase.from_yaml(
    "configs/geometry_parametric.yaml",
    "configs/fluid_selected.yaml",
)
model = PINN.load("trained/seed-42")
points_xy_m = np.array([[0.5, 0.0], [2.0, 0.0]])
result = model.predict(
    geometry=case.geometry,
    points_xy=points_xy_m,
    inlet_vmax_m_s=12.0,
    inlet_pressure_pa=180000.0,
)
print(result.u_m_s, result.v_m_s, result.p_pa)
```

The inlet pressure is the mean static pressure at the inlet section and fixes only the additive pressure gauge. The package computes

`p = p_in + rho * Vmax**2 * (pi - mean_inlet(pi))`.

Queries outside the channel or trained speed/angle range fail explicitly. `extrapolation_override=True` permits an intentional speed or angle extrapolation; different fixed channel dimensions always require retraining.

## CLI inference and export

```bash
.venv/bin/pinn-vanila predict \
  --model trained/seed-42 \
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
  --model trained/seed-42 \
  --geometry configs/geometry_parametric.yaml \
  --reference reference.csv --vmax 12 --pressure 180000 \
  --output validation-report.json
```

The report aligns the pressure gauge, keeps velocity and pressure-drop errors separate, and adds mass balance, wall slip, outlet backflow, outlet-gradient, and PDE-residual diagnostics.

## Validation boundary

The combined zero normal gradients for outlet pressure and velocity are an engineering closure, not a universal well-posed open boundary. Poiseuille flow has `dp/dx = -8 μ Vmax/H²`; its exact benchmark therefore uses that compatible nonzero outlet gradient. High-Re stationary solutions and the provisional 5% velocity/pressure-drop targets must be demonstrated against independent CFD for each `(Re, angle)` region rather than inferred from training loss.
