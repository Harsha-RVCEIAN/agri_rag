"""
Document transformers registry.

Each transformer handles ONE document class.
NO auto-discovery.
NO wildcard imports.
Explicit is mandatory to prevent leakage.
"""

from ingestion.transformers.class_a_crop import transform_class_a
from ingestion.transformers.class_b_disease import transform_class_b
from ingestion.transformers.class_c_scheme import transform_class_c
from ingestion.transformers.class_e_stats import transform_class_e

__all__ = [
    "transform_class_a",
    "transform_class_b",
    "transform_class_c",
    "transform_class_e",
]
