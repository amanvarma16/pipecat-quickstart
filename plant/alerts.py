from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .state import PlantSummary


@dataclass(slots=True)
class AlertEvent:
    key: str
    severity: str  # info|warn|urgent
    message: str


class AlertManager:
    def __init__(
        self,
        *,
        humidity_low_pct: float = 35.0,
        temp_hot_c: float = 30.0,
        vpd_high_kpa: float = 1.8,
        co2_stale_ppm: float = 400.0,
        # ROC thresholds (per minute)
        co2_spike_ppm_per_min: float = 150.0,
        humidity_drop_pct_per_min: float = 4.0,
        temp_rise_c_per_min: float = 2.0,
        # cooldowns
        cooldown_seconds: int = 45,
    ) -> None:
        self.humidity_low_pct = humidity_low_pct
        self.temp_hot_c = temp_hot_c
        self.vpd_high_kpa = vpd_high_kpa
        self.co2_stale_ppm = co2_stale_ppm
        self.co2_spike_ppm_per_min = co2_spike_ppm_per_min
        self.humidity_drop_pct_per_min = humidity_drop_pct_per_min
        self.temp_rise_c_per_min = temp_rise_c_per_min
        self.cooldown = timedelta(seconds=cooldown_seconds)

        self._last_sent: Dict[str, datetime] = {}

    def _can_send(self, key: str) -> bool:
        now = datetime.now(timezone.utc)
        last = self._last_sent.get(key)
        if not last:
            self._last_sent[key] = now
            return True
        if now - last >= self.cooldown:
            self._last_sent[key] = now
            return True
        return False

    def _dram(self, dramatic: bool, normal: str, dramatic_text: str) -> str:
        return dramatic_text if dramatic else normal

    def check(self, summary: PlantSummary, *, dramatic: bool = False) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        if not summary.latest:
            return events

        t = summary.latest.temperature_c
        h = summary.latest.humidity_pct
        c = summary.latest.co2_ppm

        # Hair dryer / drying: fast drop in humidity or rise in temp, or VPD high
        dryer_like = False
        if summary.humidity_trend_pct_per_min is not None and summary.humidity_trend_pct_per_min < -self.humidity_drop_pct_per_min:
            dryer_like = True
        if summary.temperature_trend_c_per_min is not None and summary.temperature_trend_c_per_min > self.temp_rise_c_per_min:
            dryer_like = True
        if summary.vpd_kpa is not None and summary.vpd_kpa >= self.vpd_high_kpa:
            dryer_like = True

        if dryer_like or (h <= self.humidity_low_pct and t >= self.temp_hot_c):
            key = "dry_hot"
            if self._can_send(key):
                msg = self._dram(
                    dramatic,
                    normal=f"Heads up: I'm getting dry and warm (humidity {h:.0f}%, temp {t:.0f}°C). A little water or a cooler breeze would help.",
                    dramatic_text=f"Oh no! I'm getting blasted—so dry and toasty! (humidity {h:.0f}%, temp {t:.0f}°C). Water me, stat!",
                )
                events.append(AlertEvent(key=key, severity="urgent", message=msg))

        # CO2 spike: fast rising or stale
        co2_spike = False
        if summary.co2_trend_ppm_per_min is not None and summary.co2_trend_ppm_per_min > self.co2_spike_ppm_per_min:
            co2_spike = True
        if c >= self.co2_stale_ppm:
            co2_spike = True

        if co2_spike:
            key = "co2_spike"
            if self._can_send(key):
                msg = self._dram(
                    dramatic,
                    normal=f"Air feels stale—CO2 rising (≈{c:.0f} ppm). Could you open a window?",
                    dramatic_text=f"Gasp! CO2 is soaring (≈{c:.0f} ppm)! Fresh air, please—crack a window!",
                )
                events.append(AlertEvent(key=key, severity="warn", message=msg))

        return events 