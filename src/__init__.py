"""
Billboard Analysis System Package
"""

from .production_ml_utils import ProductionBillboardAnalyzer

__version__ = "1.0.0"
__author__ = "Billboard Analysis Team"
__email__ = "team@billboard-analysis.com"

__all__ = ["ProductionBillboardAnalyzer"]

# Alias for easier import
BillboardAnalyzer = ProductionBillboardAnalyzer
