# RANS with the Wilcox (1988) k-omega closure

The default config (`configs/pinn_vanila_config.yaml`) now solves steady,
incompressible 2D mean-flow RANS. `configs/pinn_laminar_config.yaml` retains the
previous laminar Navier–Stokes workflow. This implementation uses the standard
high-Re coefficient set of the Wilcox-1988 k-omega model. It is not k-omega SST,
a transition model, LES, or an explicit resolution of turbulent fluctuations.

## Equations and conventions

Use `x*=x/H`, `U*=U/Vmax`, `p*=p/(rho Vmax²)`, `k*=k/Vmax²`,
`omega*=omega H/Vmax`, `nu_t*=nu_t/(Vmax H)`, `Re=Vmax H/nu`.
Below, all quantities and derivatives are dimensionless. The five outputs are
`u, v, p, k, omega`, with `nu_t=k/omega`, and `nu=1/Re`.

```
div U = 0
U . grad U + grad p + (2/3) grad k
    - div[(nu + nu_t)(grad U + grad U^T)] = 0

P_k = 2 nu_t S:S,   S = (grad U + grad U^T)/2

U . grad k - div[(nu + sigma_k nu_t) grad k]
    - P_k + beta_star k omega = 0

U . grad omega - div[(nu + sigma_omega nu_t) grad omega]
    - alpha (omega/k) P_k + beta omega² = 0
```

The last production term is evaluated as `alpha * 2 S:S`, the algebraically
equivalent expression for `nu_t=k/omega`. This avoids division by k at a wall.
Variable diffusivities are differentiated inside the divergence. Merely replacing
`nu` by `nu+nu_t` in a Laplacian would omit terms and is not the operator used here.

Constants: `alpha=5/9`, `beta=0.075`, `beta_star=0.09`,
`sigma_k=sigma_omega=0.5`. No SST blending, strain limiter, or low-Re damping
functions are included. The model definition follows D. C. Wilcox,
*Reassessment of the Scale-Determining Equation for Advanced Turbulence Models*,
AIAA Journal 26(11), 1988, DOI: [10.2514/3.10041](https://doi.org/10.2514/3.10041).

Pressure remains **static mean pressure**. The isotropic Reynolds stress is NOT
absorbed into it. The original inlet-mean pressure gauge is consequently unchanged:
`p = p_in + rho Vmax² (pi - mean_inlet(pi))`.

## Inlet and walls

The mean inlet parabola and wall no-slip remain prescribed by the original task.
That parabola is a chosen inflow condition, not a claim of fully developed
turbulent-channel flow. Turbulence intensity defaults to `I=0.05`, relative to
Vmax, and length scale to `ell/H=0.07`; these are explicit modeling assumptions,
not measured properties of the supplied case.
Intensity, length scale and wall regularization are fixed for each trained
checkpoint; they are not additional neural inputs. Changing them requires
new training and cannot be supplied as an arbitrary inference override.

At the inlet:

```
k_ref = 1.5 I²
k_in(eta) = k_ref (1-eta²)²
omega_core = sqrt(k_ref) / (beta_star^(1/4) (ell/H))
```

Let `d_-=(1+eta)/2` and `d_+=(1-eta)/2`, the dimensionless distances to the two
walls, and `delta=wall_omega_distance_over_width` (default 0.01). The compatible
inlet/wall omega profile is

```
omega_base = omega_core + 6/(beta Re) * [
    1/(d_-+delta)² + 1/(d_++delta)² - 2/(0.5+delta)²
]
```

It is smooth across the centerline, has the classical `6 nu/(beta d²)` near-wall
form outside the regularization region, and is finite on the walls. This is an
explicit finite-distance regularization of the singular omega condition, **not**
a calibrated wall function, a y+ constraint, or proof of mesh-independent wall
accuracy. Results must be checked for sensitivity to delta. In particular, fixed
delta is not guaranteed suitable throughout the full Reynolds range.

Network transforms enforce `k>=0` (strictly positive inside) and `omega>0`,
`k=0` on the walls, and the inlet/regularized-wall values exactly. They use
normalized softplus factors equal to one at the inlet; the omega correction also
vanishes on the walls. A smooth wall-proximity feature `(1-eta²)/4` is used in the
RANS network to avoid an absolute-value cusp in second derivatives at eta=0.

At the outlet, k and omega receive separate soft zero-normal-gradient losses.
The existing pressure/velocity zero-gradient closure is retained; its limitations,
backflow, and outlet-length sensitivity still require validation.

## Losses, diagnostics, and outputs

Transport losses use `mean((r_k/k_ref)²)` and
`mean((r_omega/omega_core²)²)`. Outlet-gradient losses use k_ref and omega_core,
respectively. These constant scalings are explicit and recorded with the model;
they do not change the zero-residual PDEs. Reports retain raw and scaled RMS
transport residuals separately. RAR includes both scaled transport residuals.

Console logs and CSV add `turbulent_k`, `turbulent_omega`, `outlet_k_gradient`,
and `outlet_omega_gradient` losses and weights. JSON diagnostics include their
raw/scaled residuals, positivity checks, and maximum eddy/molecular viscosity
ratio. Structural checks independently verify inlet k/omega and wall k/omega.
Field images and training plots include the new turbulence variables every 200
Adam steps and at the end; final fields are refreshed after L-BFGS polishing.

`PINN.predict` additionally returns:

- `k_m2_s2 = Vmax² k*`
- `omega_s_inv = Vmax/H omega*`
- `nu_t_m2_s = Vmax H nu_t* = k/omega`

CSV, HDF5 and VTK/VTU exporters include these fields for a RANS model; laminar
models retain their previous outputs. Coefficients, boundary settings and model
type are persisted. Legacy three-output checkpoints load as laminar models;
they cannot be used as five-output RANS checkpoints and require new training.

## Running and validating

Open `src/pinn_vanila/train.py` and choose **Run Python File** for GPU training.
The default output is `trained/rans-komega/seed-42`, separate from old laminar
checkpoints. CUDA availability remains required unless `--device cpu` is explicit.

```bash
.venv/bin/python src/pinn_vanila/train.py \
  --device cpu --quick --output /tmp/pinn-rans-smoke

.venv/bin/pinn-vanila-train \
  --config configs/pinn_laminar_config.yaml --output trained/laminar/seed-42
```

Operator tests compare against analytically differentiated shear, extension and
variable-omega fields. They check variable-viscosity stresses, the static-pressure
convention, production factors, both diffusion terms, physical-coordinate scaling,
and the zero-eddy-viscosity momentum limit. These are implementation checks, not
CFD accuracy validation. Fully developed Poiseuille is a laminar momentum test;
it is not an exact solution of all five positive-turbulence RANS equations.

New training and independent CFD references with the same closure, constants,
inlet profiles and wall/outlet treatment are required. Low-Re transition or
laminarization and the existence of a physically appropriate stationary solution
are not established by selecting this fully turbulent closure.
