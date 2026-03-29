from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DSEOutput:
    estimated_state: Dict[str, Any]
    context: Dict[str, Any]


class DSEAgent:
    """Data & State Estimation Agent.

    Responsibilities from paper:
    - collect local measurements and operating signals
    - perform state estimation / anomaly cleansing
    - update short-horizon operating context
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    @staticmethod
    def _smooth(values, alpha: float = 0.4):
        if not values:
            return []
        out = [values[0]]
        for v in values[1:]:
            out.append(alpha * v + (1.0 - alpha) * out[-1])
        return out

    def update(self, pdn_id: int, series, current_state=None):
        loads = [t.load for t in series]
        pvs = [t.pv for t in series]
        winds = [t.wind for t in series]
        trips = [t.trip_energy for t in series]
        buy_prices = [t.buy_price for t in series]
        sell_prices = [t.sell_price for t in series]
        freqs = [t.frequency for t in series]
        volts = [t.voltage for t in series]
        net = [l - p - w for l, p, w in zip(loads, pvs, winds)]
        volatility = max(net) - min(net) if net else 0.0
        anomalies = []
        if any(abs(f - 50.0) > 0.08 for f in freqs):
            anomalies.append('frequency_deviation')
        if any(v < 0.95 or v > 1.05 for v in volts):
            anomalies.append('voltage_out_of_band')
        if volatility > 4.0:
            anomalies.append('net_load_swing')
        mobility_pressure = sum(trips) / max(len(trips), 1)
        renewable_share = sum(pvs) + sum(winds)
        renewable_share /= max(sum(loads), 1e-6)
        risk_score = 0.0
        risk_score += min(1.0, volatility / 6.0)
        risk_score += 0.8 if 'voltage_out_of_band' in anomalies else 0.0
        risk_score += min(0.8, mobility_pressure / 0.45)
        risk_score += 0.35 if renewable_share > 0.85 else 0.0
        if current_state is not None:
            risk_score += 0.3 * current_state.risk_budget
        if risk_score < 0.75:
            risk_label = 'low'
        elif risk_score < 1.4:
            risk_label = 'normal'
        elif risk_score < 2.1:
            risk_label = 'elevated'
        else:
            risk_label = 'critical'

        estimated = {
            'load_forecast': self._smooth(loads, alpha=0.55),
            'pv_forecast': self._smooth(pvs, alpha=0.45),
            'wind_forecast': self._smooth(winds, alpha=0.45),
            'trip_energy': self._smooth(trips, alpha=0.6),
        }
        context = {
            'buy_price': buy_prices,
            'sell_price': sell_prices,
            'net_load_forecast': self._smooth(net, alpha=0.55),
            'risk_label': risk_label,
            'volatility': volatility,
            'renewable_share': renewable_share,
            'mobility_pressure': mobility_pressure,
            'anomalies': anomalies,
            'pdn_id': pdn_id,
        }
        return DSEOutput(estimated_state=estimated, context=context)
