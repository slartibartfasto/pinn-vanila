"""Prediction exporters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model import PredictionResult


def export_prediction(
    path: str | Path, points_xy: np.ndarray, result: PredictionResult
) -> None:
    destination = Path(path)
    points = np.asarray(points_xy, dtype=float)
    suffix = destination.suffix.lower()
    turbulence_fields = result.turbulence_fields()
    columns = np.column_stack(
        (
            points,
            result.u_m_s,
            result.v_m_s,
            result.p_pa,
            result.velocity_magnitude_m_s,
            *turbulence_fields.values(),
        )
    )
    if suffix == ".csv":
        np.savetxt(
            destination,
            columns,
            delimiter=",",
            header=",".join(["x_m", "y_m", "u_m_s", "v_m_s", "p_pa", "velocity_magnitude_m_s", *turbulence_fields]),
            comments="",
        )
        return
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("HDF5 export requires the optional 'h5py' package") from exc
        with h5py.File(destination, "w") as handle:
            handle.create_dataset("points_xy_m", data=points)
            handle.create_dataset("u_m_s", data=result.u_m_s)
            handle.create_dataset("v_m_s", data=result.v_m_s)
            handle.create_dataset("p_pa", data=result.p_pa)
            handle.create_dataset("velocity_magnitude_m_s", data=result.velocity_magnitude_m_s)
            for name, values in turbulence_fields.items():
                handle.create_dataset(name, data=values)
        return
    if suffix in {".vtu", ".vtk"}:
        try:
            import meshio
        except ImportError as exc:
            raise RuntimeError("VTK export requires the optional 'meshio' package") from exc
        points_3d = np.column_stack((points, np.zeros(len(points))))
        cells = [("vertex", np.arange(len(points), dtype=np.int64).reshape(-1, 1))]
        meshio.write_points_cells(
            destination,
            points_3d,
            cells,
            point_data={
                "velocity_m_s": np.column_stack(
                    (result.u_m_s, result.v_m_s, np.zeros(len(result)))
                ),
                "pressure_pa": result.p_pa,
                "velocity_magnitude_m_s": result.velocity_magnitude_m_s,
                **turbulence_fields,
            },
        )
        return
    raise ValueError("Output extension must be .csv, .h5/.hdf5, .vtu, or .vtk")
