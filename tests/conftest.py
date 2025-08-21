# Pytest global fixtures re-exported here for auto-discovery
# This allows tests to use fixtures like `init_gs` and `test_device`
# without importing them explicitly.

from .global_test_config import *  # re-export fixtures
