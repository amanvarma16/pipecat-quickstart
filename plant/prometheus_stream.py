from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

import aiohttp
from loguru import logger

from .metrics_client import PlantMetricsSample


class PrometheusPoller:
    def __init__(
        self,
        base_url: str,
        session: aiohttp.ClientSession,
        on_sample: Callable[[PlantMetricsSample], Awaitable[None]],
        *,
        interval_seconds: float = 2.0,
        ambient_co2_baseline_ppm: float = 600.0,
        temp_query: str = "temperature",
        humidity_query: str = "humidity",
        co2_query: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session
        self._on_sample = on_sample
        self._interval_seconds = interval_seconds
        self._ambient_co2_baseline_ppm = ambient_co2_baseline_ppm
        self._temp_query = temp_query
        self._humidity_query = humidity_query
        self._co2_query = co2_query
        self._stop_event = asyncio.Event()

    async def stop(self) -> None:
        self._stop_event.set()

    async def run_forever(self) -> None:
        logger.info(
            f"PrometheusPoller: starting poller at {self._base_url} interval={self._interval_seconds}s"
        )
        try:
            while not self._stop_event.is_set():
                try:
                    temp_task = asyncio.create_task(self._instant(self._temp_query))
                    hum_task = asyncio.create_task(self._instant(self._humidity_query))
                    co2_task = (
                        asyncio.create_task(self._instant(self._co2_query))
                        if self._co2_query
                        else None
                    )
                    temp, hum = await asyncio.gather(temp_task, hum_task)
                    co2 = await co2_task if co2_task else None
                    if temp is None or hum is None:
                        await self._wait_with_cancel(self._interval_seconds)
                        continue
                    co2_value = (
                        co2 if co2 is not None else self._ambient_co2_baseline_ppm
                    )
                    sample = PlantMetricsSample(
                        timestamp=datetime.now(timezone.utc),
                        co2_ppm=float(co2_value),
                        temperature_c=float(temp),
                        humidity_pct=float(hum),
                    )
                    try:
                        await self._on_sample(sample)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"PrometheusPoller: on_sample failed: {e}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"PrometheusPoller: polling error: {e}")
                await self._wait_with_cancel(self._interval_seconds)
        finally:
            logger.info("PrometheusPoller: stopped")

    async def _instant(self, query: Optional[str]) -> Optional[float]:
        if not query:
            return None
        url = f"{self._base_url}/api/v1/query"
        params = {"query": query}
        try:
            async with self._session.get(url, params=params, timeout=8) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
            result = data.get("data", {}).get("result", [])
            if not result:
                return None
            value = result[0].get("value")
            if not value or len(value) < 2:
                return None
            return float(value[1])
        except Exception:
            return None

    async def _wait_with_cancel(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return 