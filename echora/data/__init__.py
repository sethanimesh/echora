"""Compatibility access to the bundled shared lexicon, without a second copy."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("communication.backend.hinglish_core.data")
