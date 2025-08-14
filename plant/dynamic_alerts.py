from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from loguru import logger

from .alerts import AlertEvent
from .metrics_client import PlantMetricsSample


@dataclass(slots=True)
class BaselineRanges:
    temp_min_c: float
    temp_max_c: float
    temp_mean_c: float
    humidity_min_pct: float
    humidity_max_pct: float
    humidity_mean_pct: float


class RangeAlertManager:
    def __init__(
        self,
        baseline: BaselineRanges,
        *,
        temp_margin_c: float = 1.0,
        humidity_margin_pct: float = 5.0,
        shift_temp_delta_c: Optional[float] = None,
        shift_humidity_delta_pct: Optional[float] = None,
        cooldown_seconds: int = 60,
    ) -> None:
        self.baseline = baseline
        self.temp_margin_c = temp_margin_c
        self.humidity_margin_pct = humidity_margin_pct
        self.shift_temp_delta_c = shift_temp_delta_c
        self.shift_humidity_delta_pct = shift_humidity_delta_pct
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self._last_sent: Dict[str, datetime] = {}

    def _can_send(self, key: str) -> bool:
        now = datetime.now(timezone.utc)
        last = self._last_sent.get(key)
        if not last or now - last >= self.cooldown:
            self._last_sent[key] = now
            return True
        return False

    def check(self, sample: PlantMetricsSample, *, dramatic: bool = False) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        t = sample.temperature_c
        h = sample.humidity_pct

        # Out-of-range checks against baseline window ± margin
        t_low = self.baseline.temp_min_c - self.temp_margin_c
        t_high = self.baseline.temp_max_c + self.temp_margin_c
        h_low = self.baseline.humidity_min_pct - self.humidity_margin_pct
        h_high = self.baseline.humidity_max_pct + self.humidity_margin_pct

        if t < t_low or t > t_high:
            key = "temp_out_of_range"
            if self._can_send(key):
                if dramatic:
                    msg = (
                        f"Yikes! Temperature drifted outside my normal range (now {t:.1f}°C; typical {self.baseline.temp_min_c:.1f}–{self.baseline.temp_max_c:.1f}°C). "
                        f"Could we adjust the environment a bit?"
                    )
                else:
                    msg = (
                        f"Temperature is outside my usual range (now {t:.1f}°C; typical {self.baseline.temp_min_c:.1f}–{self.baseline.temp_max_c:.1f}°C)."
                    )
                events.append(AlertEvent(key=key, severity="warn", message=msg))

        if h < h_low or h > h_high:
            key = "humidity_out_of_range"
            if self._can_send(key):
                if dramatic:
                    msg = (
                        f"Uh‑oh! Humidity slipped outside my comfort window (now {h:.0f}%; typical {self.baseline.humidity_min_pct:.0f}–{self.baseline.humidity_max_pct:.0f}%). "
                        f"A little adjustment would help."
                    )
                else:
                    msg = (
                        f"Humidity is outside my usual range (now {h:.0f}%; typical {self.baseline.humidity_min_pct:.0f}–{self.baseline.humidity_max_pct:.0f}%)."
                    )
                events.append(AlertEvent(key=key, severity="warn", message=msg))

        # Shift checks vs baseline mean, only if not already out-of-range
        if not any(e.key in ("temp_out_of_range",) for e in events) and self.shift_temp_delta_c is not None:
            if abs(t - self.baseline.temp_mean_c) >= self.shift_temp_delta_c:
                key = "temp_shift"
                if self._can_send(key):
                    delta = t - self.baseline.temp_mean_c
                    dir_txt = "warmer" if delta > 0 else "cooler"
                    msg = (
                        f"Temperature feels {dir_txt} than usual by {abs(delta):.1f}°C (now {t:.1f}°C; typical mean {self.baseline.temp_mean_c:.1f}°C)."
                    )
                    events.append(AlertEvent(key=key, severity="info", message=msg))

        if not any(e.key in ("humidity_out_of_range",) for e in events) and self.shift_humidity_delta_pct is not None:
            if abs(h - self.baseline.humidity_mean_pct) >= self.shift_humidity_delta_pct:
                key = "humidity_shift"
                if self._can_send(key):
                    delta = h - self.baseline.humidity_mean_pct
                    dir_txt = "higher" if delta > 0 else "lower"
                    msg = (
                        f"Humidity is {dir_txt} than usual by {abs(delta):.0f}% (now {h:.0f}%; typical mean {self.baseline.humidity_mean_pct:.0f}%)."
                    )
                    events.append(AlertEvent(key=key, severity="info", message=msg))

        return events 