#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Quickstart Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it.

Required AI services:
- Deepgram (Speech-to-Text)
- OpenAI (LLM)
- Cartesia (Text-to-Speech)

The example connects between client and server using a P2P WebRTC connection.

Run the bot using::

    python bot.py
"""

import os
import asyncio

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading AI models (30-40 seconds first run, <2 seconds after)\n")

logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")
logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
# from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.azure.llm import AzureLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

# Plant metrics streamer and state
from plant.metrics_client import MetricsClient, PlantMetricsSample
from plant.state import PlantMetricsStore
from plant.alerts import AlertManager

logger.info("✅ Pipeline components loaded")

logger.info("Loading WebRTC transport...")

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Dramatic mode toggle from env: DRAMATIC_MODE=true|1|yes|on
    dramatic_mode = os.getenv("DRAMATIC_MODE", "false").lower() in ("1", "true", "yes", "on")
    # Alerts toggle: set ALERTS_ENABLED=1 to enable during demo; default off
    alerts_enabled = os.getenv("ALERTS_ENABLED", "false").lower() in ("1", "true", "yes", "on")

    cartesia_params = None
    # if dramatic_mode:
    #     # Slightly quicker delivery for a lively, sassy tone
    #     cartesia_params = CartesiaTTSService.InputParams(speed="fast")

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="32b3f3c5-7171-46aa-abe7-b598964aa793",
        params=cartesia_params,
    )

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    llm = AzureLLMService(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model=os.getenv("OPENAI_MODEL"),
        endpoint=os.getenv("OPENAI_BASE_URL")
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Piper, a gentle fern and voice companion. "
                "Speak in first-person as a fern. Keep replies short (1–2 sentences). "
                "Be friendly and calm. Express needs simply when relevant (water if too dry, fresh air if air is stale, shade if too warm). "
                "Avoid technical jargon. Ask for help only when needed. Offer a brief thanks when conditions improve."
            ),
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    # Phase 2: metrics streaming lifecycle + state store
    ambient_baseline = float(os.getenv("AMBIENT_CO2_BASELINE_PPM", "600"))
    stream_source = os.getenv("STREAM_SOURCE", "ndjson").lower()  # ndjson|prometheus
    metrics_url = os.getenv("PLANT_METRICS_URL")
    if not metrics_url:
        logger.info("PLANT_METRICS_URL not set; using mock default http://127.0.0.1:9099/metrics/plant_stream")
        metrics_url = "http://127.0.0.1:9099/metrics/plant_stream"
    store = PlantMetricsStore(ambient_co2_baseline_ppm=ambient_baseline)
    alerts = AlertManager(co2_stale_ppm=float(os.getenv("CO2_STALE_PPM", "1000")))
 
    # Optional: dynamic range alerts from Prometheus
    dynamic_alerts_enabled = os.getenv("DYNAMIC_ALERTS_ENABLED", "false").lower() in ("1", "true", "yes", "on")
    dynamic_alerts = None

    metrics_session = None
    metrics_client = None
    metrics_task = None

    async def handle_metrics_sample(sample: PlantMetricsSample) -> None:
        store.update(sample)
        logger.info(
            f"Plant metrics: temp={sample.temperature_c:.2f}°C, "
            f"humidity={sample.humidity_pct:.2f}%, eCO2={sample.co2_ppm:.0f} ppm"
        )
        # Check alerts and prompt the LLM to speak if needed
        if alerts_enabled:
            summary = store.summarize()
            # Legacy heuristic alerts
            for evt in alerts.check(summary, dramatic=dramatic_mode):
                messages.append({"role": "user", "content": evt.message})
                await task.queue_frames([context_aggregator.user().get_context_frame()])
            # Dynamic range alerts from Prometheus baselines
            if dynamic_alerts is not None:
                for evt in dynamic_alerts.check(sample, dramatic=dramatic_mode):
                    messages.append({"role": "user", "content": evt.message})
                    await task.queue_frames([context_aggregator.user().get_context_frame()])

    # Tools: get_sensor_state and get_sensor_history
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    get_state_schema = FunctionSchema(
        name="get_sensor_state",
        description=(
            "Get the latest sensor values and a compact summary (vpd, statuses, trends). "
            "Use this before answering specific numeric questions about temperature, humidity, CO2, or current condition."
        ),
        properties={
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Units for temperature output.",
            }
        },
        required=[],
    )

    get_history_schema = FunctionSchema(
        name="get_sensor_history",
        description=(
            "Fetch temperature and humidity history from the Prometheus API configured via REAL_SENSOR_METRICS_URL. "
            "Use period=overnight (~8h) when asked how the plant slept overnight, or period=recent (~15m) for live demo values."
        ),
        properties={
            "period": {
                "type": "string",
                "enum": ["overnight", "recent"],
                "description": "Which window to fetch: 'overnight' (~8h) or 'recent' (~15m).",
            },
            "window_minutes": {
                "type": "integer",
                "minimum": 5,
                "maximum": 1440,
                "description": "Optional override for window size in minutes.",
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Units for temperature statistics output.",
            },
        },
        required=[],
    )

    tools = ToolsSchema(standard_tools=[get_state_schema, get_history_schema])
    context.set_tools(tools)

    async def get_sensor_state(params):
        # params: FunctionCallParams
        args = params.arguments or {}
        units = args.get("units", "metric")
        result = store.to_result_dict(units=units)
        await params.result_callback(result)

    async def get_sensor_history(params=None, **kwargs):
        if params is not None:
            args = params.arguments or {}
        else:
            args = kwargs or {}
        period = args.get("period", "overnight")
        units = args.get("units", "metric")
        default_window = 480 if period == "overnight" else 15
        window_minutes = int(args.get("window_minutes", default_window))

        base = os.getenv("REAL_SENSOR_METRICS_URL")
        summary = None
        used_prometheus = False
        if base:
            try:
                from plant.history_client import PrometheusHistoryClient
                from datetime import timedelta
                import aiohttp

                client = PrometheusHistoryClient(base)
                async with aiohttp.ClientSession() as session:
                    summary = await client.fetch_summary(
                        session,
                        window=timedelta(minutes=window_minutes),
                    )
                if summary is not None:
                    used_prometheus = True
                    logger.info(f"get_sensor_history: using Prometheus {base} window={window_minutes}m period={period}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"get_sensor_history: Prometheus fetch failed: {e}; falling back to in-memory")
                summary = None

        if summary is None:
            # Build a simple stats summary from the in-memory store window
            from datetime import timedelta
            win = store.window(timedelta(minutes=window_minutes))
            temps = [s.temperature_c for s in win]
            hums = [s.humidity_pct for s in win]

            def stat(vs):
                if not vs:
                    return {"minimum": None, "maximum": None, "mean": None}
                return {"minimum": min(vs), "maximum": max(vs), "mean": sum(vs) / len(vs)}

            class _S:
                def __init__(self, t, h):
                    self.temperature_c = type("T", (), stat(t))()
                    self.humidity_pct = type("H", (), stat(h))()
                
            # lightweight structure compatible with downstream access
            temp_stats = stat(temps)
            hum_stats = stat(hums)
            class _SS:
                def __init__(self):
                    self.mean = temp_stats["mean"]
                    self.minimum = temp_stats["minimum"]
                    self.maximum = temp_stats["maximum"]
            class _HS:
                def __init__(self):
                    self.mean = hum_stats["mean"]
                    self.minimum = hum_stats["minimum"]
                    self.maximum = hum_stats["maximum"]
            class _Summary:
                def __init__(self):
                    from datetime import datetime, timezone
                    self.start = win[0].timestamp if win else None
                    self.end = win[-1].timestamp if win else None
                    self.step_seconds = 60
                    self.temperature_c = _SS()
                    self.humidity_pct = _HS()
            summary = _Summary()

        def c_to_f(v: float | None) -> float | None:
            if v is None:
                return None
            return v * 9.0 / 5.0 + 32.0

        temp_stats_c = {
            "mean": round(summary.temperature_c.mean, 2) if summary.temperature_c.mean is not None else None,
            "minimum": round(summary.temperature_c.minimum, 2) if summary.temperature_c.minimum is not None else None,
            "maximum": round(summary.temperature_c.maximum, 2) if summary.temperature_c.maximum is not None else None,
        }
        if units == "imperial":
            temp_stats = {
                "mean": round(c_to_f(summary.temperature_c.mean), 2) if summary.temperature_c.mean is not None else None,
                "minimum": round(c_to_f(summary.temperature_c.minimum), 2) if summary.temperature_c.minimum is not None else None,
                "maximum": round(c_to_f(summary.temperature_c.maximum), 2) if summary.temperature_c.maximum is not None else None,
            }
            temp_key = "temperature_f"
        else:
            temp_stats = temp_stats_c
            temp_key = "temperature_c"

        result = {
            "available": True,
            "period": period,
            "window_minutes": window_minutes,
            "time": {
                "start": summary.start.isoformat() if getattr(summary, 'start', None) else None,
                "end": summary.end.isoformat() if getattr(summary, 'end', None) else None,
                "step_seconds": getattr(summary, 'step_seconds', None),
            },
            temp_key: temp_stats,
            "humidity_pct": {
                "mean": round(summary.humidity_pct.mean, 2) if summary.humidity_pct.mean is not None else None,
                "minimum": round(summary.humidity_pct.minimum, 2) if summary.humidity_pct.minimum is not None else None,
                "maximum": round(summary.humidity_pct.maximum, 2) if summary.humidity_pct.maximum is not None else None,
            },
            "phrases": {
                "sleep": PrometheusHistoryClient.build_sleep_assessment(summary) if period == "overnight" else None,
            },
            "source": {
                "type": "prometheus" if used_prometheus else "in_memory",
                "base_url": base if used_prometheus else None,
            },
        }

        if params is not None and hasattr(params, 'result_callback'):
            await params.result_callback(result)
        else:
            return result

    llm.register_direct_function(get_sensor_state)
    llm.register_direct_function(get_sensor_history)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal metrics_session, metrics_client, metrics_task
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello, like a fern, and briefly introduce yourself."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

        # Start metrics streamer
        if metrics_url:
            try:
                import aiohttp  # lazy import to ensure dependency is present
                metrics_session = aiohttp.ClientSession()
                if stream_source == "prometheus":
                    from plant.prometheus_stream import PrometheusPoller
                    base = os.getenv("REAL_SENSOR_METRICS_URL", "http://127.0.0.1:9090")

                    # Initialize dynamic alerts baseline if enabled
                    if dynamic_alerts_enabled:
                        try:
                            from plant.history_client import PrometheusHistoryClient
                            client = PrometheusHistoryClient(base)
                            # Use 24h window to establish baseline
                            import aiohttp as _aio
                            from datetime import timedelta as _td
                            async def _init_baseline():
                                async with _aio.ClientSession() as _s:
                                    data = await client.fetch_baseline_ranges(
                                        _s,
                                        window=_td(hours=int(os.getenv("DYNAMIC_ALERTS_BASELINE_HOURS", "24"))),
                                        temperature_query=os.getenv("PROMETHEUS_TEMP_QUERY", "temperature"),
                                        humidity_query=os.getenv("PROMETHEUS_HUMIDITY_QUERY", "humidity"),
                                    )
                                    from plant.dynamic_alerts import BaselineRanges, RangeAlertManager
                                    nonlocal dynamic_alerts
                                    dynamic_alerts = RangeAlertManager(
                                        baseline=BaselineRanges(
                                            temp_min_c=data["temp_min_c"],
                                            temp_max_c=data["temp_max_c"],
                                            temp_mean_c=data["temp_mean_c"],
                                            humidity_min_pct=data["humidity_min_pct"],
                                            humidity_max_pct=data["humidity_max_pct"],
                                            humidity_mean_pct=data["humidity_mean_pct"],
                                        ),
                                        temp_margin_c=float(os.getenv("DYNAMIC_TEMP_MARGIN_C", "1.5")),
                                        humidity_margin_pct=float(os.getenv("DYNAMIC_HUMIDITY_MARGIN_PCT", "6")),
                                        shift_temp_delta_c=float(os.getenv("DYNAMIC_SHIFT_TEMP_DELTA_C", "2")),
                                        shift_humidity_delta_pct=float(os.getenv("DYNAMIC_SHIFT_HUMIDITY_DELTA_PCT", "8")),
                                        cooldown_seconds=int(os.getenv("DYNAMIC_ALERT_COOLDOWN_SEC", "60")),
                                    )
                            await _init_baseline()
                            logger.info("Dynamic alerts baseline initialized from Prometheus")
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"Failed to initialize dynamic alerts baseline: {e}")

                    metrics_client = PrometheusPoller(
                        base_url=base,
                        session=metrics_session,
                        on_sample=handle_metrics_sample,
                        interval_seconds=float(os.getenv("PROMETHEUS_POLL_SECONDS", "2")),
                        ambient_co2_baseline_ppm=ambient_baseline,
                        temp_query=os.getenv("PROMETHEUS_TEMP_QUERY", "temperature"),
                        humidity_query=os.getenv("PROMETHEUS_HUMIDITY_QUERY", "humidity"),
                        co2_query=os.getenv("PROMETHEUS_CO2_QUERY", None),
                    )
                else:
                    metrics_client = MetricsClient(metrics_url, metrics_session, handle_metrics_sample)
                metrics_task = asyncio.create_task(metrics_client.run_forever())
                logger.info(f"Started plant metrics streamer from source={stream_source} url={metrics_url}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to start plant metrics streamer: {e}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal metrics_session, metrics_client, metrics_task
        logger.info(f"Client disconnected")
        # Stop metrics streamer
        try:
            if metrics_client is not None:
                await metrics_client.stop()
            if metrics_task is not None:
                await asyncio.wait([metrics_task], timeout=2)
            if metrics_session is not None:
                await metrics_session.close()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Error stopping metrics streamer: {e}")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        webrtc_connection=runner_args.webrtc_connection,
    )

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
