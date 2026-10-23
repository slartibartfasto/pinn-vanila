# Design and validation notes

The default workflow now uses RANS k-omega; see [the RANS model definition](rans.md)
for all five equations, boundary profiles, scalings and limitations. The laminar
workflow below remains available through `configs/pinn_laminar_config.yaml`.

The package follows the supplied technical specification while keeping claims separate from implementation. The architecture and training budgets are engineering hypotheses. A successful optimizer run does not establish that a unique physically relevant stationary solution exists throughout `Re=500..10000`.

The map is `r(s, eta) = r_c(s) + H eta n(s)/2`. Its minimum arc Jacobian factor is `1-H/(2R)=0.75`. The NumPy map is used to reject invalid query points; the equivalent Torch map preserves autograd from network outputs back to physical `x,y` for first and second derivatives.

Velocity is represented as a centerline-tangent parabolic base plus a learned correction multiplied by `(1-eta²) s/(s+H)`. This makes the inlet profile and wall no-slip exact. Pressure remains unconstrained until the inference layer subtracts its quadrature mean on the inlet.

The training objective records continuity, both momentum components, outlet pressure gradient, outlet velocity gradient, and flux separately. Reference labels have zero training weight. Residual adaptive refinement retains a configured uniform fraction. The model metadata reports training and validation state so an unvalidated checkpoint cannot silently appear as a validated solver.

Independent acceptance work still requires boundary-condition-matched CFD, grid convergence, three random seeds, held-out speeds and angles, outlet-length sensitivity, backflow diagnostics, and expert-overlap checks if the single network fails at high Reynolds number.
