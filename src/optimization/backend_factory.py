from __future__ import annotations

from typing import Dict, Any

from .heuristic_solver import HeuristicDispatchSolver
from .pyomo_backend import PyomoBackend
from .cvxpy_backend import CVXPYBackend
from .short_term.admm_dispatch import DistributedADMMDispatcher
from .short_term.pyomo_dispatch import PyomoShortTermDispatcher


def build_solver_backend(config: Dict[str, Any]):
    solver_cfg = config.get('solver', {})
    preferred = solver_cfg.get('preferred', 'admm')
    if preferred in ('admm', 'paper'):
        return DistributedADMMDispatcher(
            rho=solver_cfg.get('admm_rho', 1.0),
            max_iter=solver_cfg.get('admm_max_iter', 10),
            tol=solver_cfg.get('admm_tol', 1.0e-3),
        )
    if preferred == 'pyomo':
        return PyomoShortTermDispatcher(solver_name=solver_cfg.get('pyomo_solver', 'glpk'))
    if preferred == 'cvxpy':
        return CVXPYBackend()
    if preferred == 'pyomo_shell':
        return PyomoBackend(solver_name=solver_cfg.get('pyomo_solver', 'glpk'))
    return HeuristicDispatchSolver()
