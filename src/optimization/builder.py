from __future__ import annotations

from typing import Dict, Any

from .schemas import OptimizationProblem


class OptimizationProblemBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def build(self, day_data, plan, bridge, initial_state=None, dse_outputs=None, memory_context=None) -> OptimizationProblem:
        num_pdns = len(day_data)
        horizon = len(next(iter(day_data.values())))
        rolling_window_min = self.cfg.get('runtime', {}).get('rolling_window_min', 60)
        step_minutes = self.cfg.get('runtime', {}).get('step_minutes', 60)
        effective_window_steps = max(1, min(horizon, int(round(rolling_window_min / max(step_minutes, 1)))))
        use_dse = dse_outputs is not None
        forecasts = {
            'load': {n: list(dse_outputs[n].estimated_state['load_forecast']) if use_dse else [t.load for t in series] for n, series in day_data.items()},
            'pv': {n: list(dse_outputs[n].estimated_state['pv_forecast']) if use_dse else [t.pv for t in series] for n, series in day_data.items()},
            'wind': {n: list(dse_outputs[n].estimated_state['wind_forecast']) if use_dse else [t.wind for t in series] for n, series in day_data.items()},
            'trip_energy': {n: list(dse_outputs[n].estimated_state['trip_energy']) if use_dse else [t.trip_energy for t in series] for n, series in day_data.items()},
            'buy_price': {n: [t.buy_price for t in series] for n, series in day_data.items()},
            'sell_price': {n: [t.sell_price for t in series] for n, series in day_data.items()},
        }
        initial_battery_soc = {
            n: (
                initial_state[n].battery_soc
                if initial_state is not None
                else 0.6 * plan.battery_energy[n]
            )
            for n in range(num_pdns)
        }
        initial_ev_soc = {
            n: (
                initial_state[n].ev_soc
                if initial_state is not None
                else max(1.0, 0.75 * plan.v2g_ratio[n] * 4.0)
            )
            for n in range(num_pdns)
        }
        constraints = {
            'system_import_cap': self.cfg['short_term']['system_import_cap_default'],
            'battery_energy_cap': plan.battery_energy,
            'battery_power_cap': plan.battery_power,
            'export_cap': plan.export_cap,
            'grid_exchange_cap': bridge.grid_exchange_cap,
            'reserve_requirement': bridge.reserve_requirement,
            'initial_battery_soc': initial_battery_soc,
            'initial_ev_soc': initial_ev_soc,
            'ev_energy_cap': {n: max(2.0, plan.v2g_ratio[n] * 4.0) for n in range(num_pdns)},
            'ev_power_cap': {n: max(1.0, plan.v2g_ratio[n] * 2.0) for n in range(num_pdns)},
            'ev_energy_requirement': bridge.ev_energy_requirement,
            'terminal_soc': bridge.terminal_soc,
            'risk_budget': bridge.risk_budget,
            'effective_window_steps': effective_window_steps,
        }
        return OptimizationProblem(
            horizon=horizon,
            num_pdns=num_pdns,
            objective_weights=self.cfg['costs'],
            parameters={
                'plan': plan,
                'bridge': bridge,
                'battery_charge_eff': self.cfg['short_term']['battery_charge_eff'],
                'battery_discharge_eff': self.cfg['short_term']['battery_discharge_eff'],
                'ev_charge_eff': self.cfg['short_term']['ev_charge_eff'],
                'ev_discharge_eff': self.cfg['short_term']['ev_discharge_eff'],
                'rolling_window_min': rolling_window_min,
                'step_minutes': step_minutes,
                'admm_rho': self.cfg.get('solver', {}).get('admm_rho', 1.0),
                'admm_max_iter': self.cfg.get('solver', {}).get('admm_max_iter', 10),
                'admm_tol': self.cfg.get('solver', {}).get('admm_tol', 1.0e-3),
                'warm_start_hint': (memory_context or {}).get('checkpoint', {}).get('warm_start', {}),
            },
            forecasts=forecasts,
            constraints=constraints,
        )
