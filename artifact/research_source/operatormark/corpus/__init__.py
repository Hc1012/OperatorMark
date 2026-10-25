"""Guarded private-corpus tooling.

Importing this package performs no numerical import, RNG construction, solver
execution, filesystem write, or state transition.  Submodules are imported
only after the parent launcher and child pre-import checks have succeeded.
"""
