"""
Clinical Trial Protocol Review Environment for OpenEnv.

Usage:
    from openenv import AutoEnv, AutoAction
    env = AutoEnv.from_env("clinical-trial-env")
    action = AutoAction.from_env("clinical-trial-env")

Or directly:
    from clinical_trial_env import ClinicalTrialEnv, ClinicalTrialAction
"""

from .models import (
    ClinicalTrialAction,
    ClinicalTrialObservation,
    ViolationFlag,
    Severity,
)
from .client import ClinicalTrialEnv

# OpenEnv auto-discovery expects these exact names
# AutoEnv looks for: <PackageName>Env class
# AutoAction looks for: <PackageName>Action class
__env_class__ = ClinicalTrialEnv
__action_class__ = ClinicalTrialAction

__all__ = [
    "ClinicalTrialEnv",
    "ClinicalTrialAction",
    "ClinicalTrialObservation",
    "ViolationFlag",
    "Severity",
    "__env_class__",
    "__action_class__",
]
