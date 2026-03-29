from __future__ import annotations

from .schemas import OptimizationProblem, OptimizationResult
from .solver_base import SolverBackend


class CVXPYBackend(SolverBackend):
    name = 'cvxpy'

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        try:
            import cvxpy as cp  # type: ignore
        except Exception as exc:
            return OptimizationResult(
                status='backend_unavailable',
                objective_value=0.0,
                variables={},
                diagnostics={'backend': 'cvxpy', 'error': f'CVXPY import failed: {exc}'},
            )
        x = cp.Variable(nonneg=True)
        objective = cp.Minimize(x)
        constraints = [x >= 0]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
        except Exception as exc:
            return OptimizationResult(
                status='solve_error',
                objective_value=0.0,
                variables={},
                diagnostics={'backend': 'cvxpy', 'error': str(exc)},
            )
        return OptimizationResult(
            status='placeholder_solved',
            objective_value=float(prob.value or 0.0),
            variables={'x': float(x.value or 0.0)},
            diagnostics={'backend': 'cvxpy', 'note': 'Placeholder convex backend; full dispatch model not yet encoded.'},
        )
