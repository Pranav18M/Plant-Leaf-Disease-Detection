"""
Plant Leaf Disease Detection System
Source Package Initialization
"""

__version__ = "1.0.0"
__author__ = "Plant Disease Detection Team"
__description__ = "Automated plant leaf disease detection using classical machine learning"

# Make modules easily accessible
from . import preprocessing
from . import feature_extraction
from . import utils

__all__ = ['preprocessing', 'feature_extraction', 'utils']