"""Compatibility import of the single authoritative language core."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module('communication.backend.hinglish_core.core.model')
