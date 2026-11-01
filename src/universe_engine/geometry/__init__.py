"""Higher-dimensional geometry core."""
from .geometry_core import (
    CoordinateChart, MetricSignature, MetricTensor, ChristoffelSymbols,
    Curvature, Field, BoundaryCondition, GeometrySpec,
    compute_christoffel, compute_curvature_from_metric,
    ConstantMetric, FLRWMetric, Bulk5DMetric,
    flat_minkowski_4d, flat_minkowski_5d, circle_compactification,
)

__all__ = [
    "CoordinateChart", "MetricSignature", "MetricTensor", "ChristoffelSymbols",
    "Curvature", "Field", "BoundaryCondition", "GeometrySpec",
    "compute_christoffel", "compute_curvature_from_metric",
    "ConstantMetric", "FLRWMetric", "Bulk5DMetric",
    "flat_minkowski_4d", "flat_minkowski_5d", "circle_compactification",
]
