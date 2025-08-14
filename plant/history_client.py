from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp
from loguru import logger


@dataclass(slots=True)
class SeriesPoint:
    timestamp: datetime
    value: float


@dataclass(slots=True)
class SeriesStats:
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]


@dataclass(slots=True)
class HistorySummary:
    start: datetime
    end: datetime
    step_seconds: int
    temperature_c: SeriesStats
    humidity_pct: SeriesStats


class PrometheusHistoryClient:
    def __init__(self, base_url: str) -> None:
        # Example: http://127.0.0.1:9090
        self._base_url = base_url.rstrip("/")

    async def _fetch_series(
        self,
        session: aiohttp.ClientSession,
        query: str,
        start: datetime,
        end: datetime,
        step_seconds: int,
    ) -> List[SeriesPoint]:
        url = f"{self._base_url}/api/v1/query_range"
        params = {
            "query": query,
            "start": f"{start.timestamp():.0f}",
            "end": f"{end.timestamp():.0f}",
            "step": str(step_seconds),
        }
        try:
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                data = await resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"PrometheusHistoryClient: failed to fetch {query}: {e}")
            return []

        try:
            result = data.get("data", {}).get("result", [])
            if not result:
                return []
            # Take the first series
            values = result[0].get("values", [])
            points: List[SeriesPoint] = []
            for ts, val in values:
                try:
                    # Prometheus returns ts as float seconds (or string) and val as string
                    tsf = float(ts)
                    vf = float(val)
                    if vf != vf:  # NaN check
                        continue
                    points.append(
                        SeriesPoint(timestamp=datetime.fromtimestamp(tsf, tz=timezone.utc), value=vf)
                    )
                except Exception:
                    continue
            return points
        except Exception as e:  # noqa: BLE001
            logger.warning(f"PrometheusHistoryClient: failed parsing response for {query}: {e}")
            return []

    @staticmethod
    def _stats(points: List[SeriesPoint]) -> SeriesStats:
        if not points:
            return SeriesStats(minimum=None, maximum=None, mean=None)
        vals = [p.value for p in points]
        return SeriesStats(minimum=min(vals), maximum=max(vals), mean=sum(vals) / len(vals))

    async def fetch_summary(
        self,
        session: aiohttp.ClientSession,
        *,
        window: timedelta,
        step_seconds: int = 60,
        temperature_query: str = "temperature",
        humidity_query: str = "humidity",
    ) -> HistorySummary:
        end = datetime.now(timezone.utc)
        start = end - window
        temp_task = asyncio.create_task(
            self._fetch_series(session, temperature_query, start, end, step_seconds)
        )
        hum_task = asyncio.create_task(
            self._fetch_series(session, humidity_query, start, end, step_seconds)
        )
        temperature_series, humidity_series = await asyncio.gather(temp_task, hum_task)
        return HistorySummary(
            start=start,
            end=end,
            step_seconds=step_seconds,
            temperature_c=self._stats(temperature_series),
            humidity_pct=self._stats(humidity_series),
        )

    @staticmethod
    def build_sleep_assessment(summary: HistorySummary) -> str:
        t = summary.temperature_c.mean
        h = summary.humidity_pct.mean
        if t is None or h is None:
            return "I rested okay. I don't have complete overnight data, but things seemed calm."
        if 18.0 <= t <= 28.0 and 35.0 <= h <= 80.0:
            return "I slept well—comfortable temperature and humidity through the night."
        if t > 30.0 and h < 35.0:
            return "It was warm and dry overnight—I could use a little water and cooler air."
        if t > 30.0:
            return "It was a bit warm overnight—some shade or a cooler breeze would help."
        if h < 35.0:
            return "Overnight felt dry—a small drink of water would be lovely."
        return "I slept fine overall—thanks for checking on me!"

    async def fetch_baseline_ranges(
        self,
        session: aiohttp.ClientSession,
        *,
        window: timedelta,
        step_seconds: int = 60,
        temperature_query: str = "temperature",
        humidity_query: str = "humidity",
    ) -> Dict[str, float]:
        summary = await self.fetch_summary(
            session,
            window=window,
            step_seconds=step_seconds,
            temperature_query=temperature_query,
            humidity_query=humidity_query,
        )
        return {
            "temp_min_c": summary.temperature_c.minimum or 0.0,
            "temp_max_c": summary.temperature_c.maximum or 0.0,
            "temp_mean_c": summary.temperature_c.mean or 0.0,
            "humidity_min_pct": summary.humidity_pct.minimum or 0.0,
            "humidity_max_pct": summary.humidity_pct.maximum or 0.0,
            "humidity_mean_pct": summary.humidity_pct.mean or 0.0,
        } 