"""Run all tests.

Author: Matthew Harrigan
"""
from future.builtins import *
import unittest
from pkg_resources import resource_filename
import sys


def main(verbosity=1):
    """Discover and run tests."""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    suite = unittest.defaultTestLoader.discover(
        resource_filename('wetmsm', 'testing/'))
    return runner.run(suite)


if __name__ == "__main__":
    ret = main(verbosity=2)
    if ret.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)
