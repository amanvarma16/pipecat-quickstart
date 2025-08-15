# Plant Chat 🌱

A voice AI plant companion that monitors environmental conditions and provides intelligent alerts. This bot uses Pipecat to create a conversational interface for your plants, allowing you to talk to them and get real-time feedback about their environment.

## Prerequisites

### Python 3.10+

Pipecat requires Python 3.10 or newer. Check your version:

```bash
python --version
```

If you need to upgrade Python, we recommend using a version manager like `uv` or `pyenv`.

### AI Service API keys

Pipecat orchestrates different AI services in a pipeline, ensuring low latency communication. Plant Chat uses:

- [Deepgram](https://console.deepgram.com/signup) for Speech-to-Text transcriptions
- [OpenAI](https://auth.openai.com/create-account) or [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) for LLM inference
- [Cartesia](https://play.cartesia.ai/sign-up) for Text-to-Speech audio generation

Have your API keys ready. We'll add them to your `.env` shortly.

### Optional: Plant Sensors

For the full Plant Chat experience, you can connect real plant sensors:
- **Prometheus-compatible metrics** (temperature, humidity, CO2)
- **Mock data stream** for testing
- **Real-time environmental monitoring**

## Setup

1. Clone this repository

   ```bash
   git clone https://github.com/amanvarma16/plant-chat.git
   cd plant-chat
   ```

2. Configure environment variables

   Create a `.env` file:

   ```bash
   cp env.example .env
   ```

   Then, add your API keys:

   ```
   # Required AI Services
   DEEPGRAM_API_KEY=your_deepgram_api_key
   OPENAI_API_KEY=your_openai_api_key
   CARTESIA_API_KEY=your_cartesia_api_key
   
   # Optional: Azure OpenAI (if using instead of OpenAI)
   OPENAI_BASE_URL=https://your-resource.openai.azure.com/
   OPENAI_MODEL=gpt-4
   
   # Optional: Plant Sensors
   PLANT_METRICS_URL=http://127.0.0.1:9099/metrics/plant_stream
   REAL_SENSOR_METRICS_URL=http://127.0.0.1:9090
   ```

3. Set up a virtual environment and install dependencies

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   > Using `uv`? Create your venv using: `uv sync`

4. Run the example

   ```bash
   python bot.py
   ```

   > Using `uv`? Run your bot using: `uv run bot.py`.

5. Connect and test

   **Open http://localhost:7860/client in your browser** and click `Connect` to start talking to your plant companion.

   > 💡 First run note: The initial startup may take ~15 seconds as Pipecat downloads required models, like the Silero VAD model.

## What Plant Chat Does 🌿

Plant Chat transforms your plants into conversational companions that can:

### 🎤 **Talk to You**
- Respond to voice conversations naturally
- Express their needs and feelings about the environment
- Provide updates on their current condition

### 📊 **Monitor Environment**
- Track temperature, humidity, and CO2 levels in real-time
- Detect environmental changes and trends
- Assess overnight conditions and "sleep quality"

### 🚨 **Smart Alerts**
- **Dynamic Alerts**: Learn your plant's normal ranges and alert when conditions drift outside them
- **Trend Alerts**: Detect rapid changes (like a hair dryer effect)
- **CO2 Monitoring**: Alert when air becomes stale or CO2 spikes
- **VPD Analysis**: Monitor vapor pressure deficit for optimal growth

### 🛠 **Built-in Tools**
- `get_sensor_state`: Get current readings and status
- `get_sensor_history`: Fetch historical data and trends
- Sleep assessment for overnight conditions

## Configuration Options

### Alert System
```bash
# Enable alerts (default: false)
ALERTS_ENABLED=true

# Enable dynamic baseline alerts (requires Prometheus)
DYNAMIC_ALERTS_ENABLED=true

# Alert sensitivity
DYNAMIC_TEMP_MARGIN_C=1.5
DYNAMIC_HUMIDITY_MARGIN_PCT=6
DYNAMIC_SHIFT_TEMP_DELTA_C=2
DYNAMIC_SHIFT_HUMIDITY_DELTA_PCT=8
```

### Personality
```bash
# Dramatic mode for more expressive responses
DRAMATIC_MODE=true
```

### Data Sources
```bash
# Use Prometheus for real sensor data
STREAM_SOURCE=prometheus

# Use mock data for testing
STREAM_SOURCE=ndjson
```

## Troubleshooting

- **Browser permissions**: Make sure to allow microphone access when prompted by your browser.
- **Connection issues**: If the WebRTC connection fails, first try a different browser. If that fails, make sure you don't have a VPN or firewall rules blocking traffic. WebRTC uses UDP to communicate.
- **Audio issues**: Check that your microphone and speakers are working and not muted.
- **Sensor data**: If using real sensors, ensure Prometheus is running and accessible at the configured URL.
- **Dynamic alerts**: Make sure `DYNAMIC_ALERTS_ENABLED=true` and `ALERTS_ENABLED=true` for intelligent alerts.

## Example Conversations

Try asking your plant companion:

- "How are you feeling today?"
- "What's the temperature like?"
- "Did you sleep well last night?"
- "Are you getting enough water?"
- "Is the air fresh around you?"
- "What's your current environment like?"

## Next Steps

- **Connect real sensors**: Set up Prometheus-compatible sensors for live environmental monitoring
- **Customize alerts**: Adjust alert thresholds and sensitivity for your specific plants
- **Extend functionality**: Add more sensor types or custom plant personalities
- **Read the docs**: Check out [Pipecat's docs](https://docs.pipecat.ai/) for guides and reference information.
- **Join Discord**: Join [Pipecat's Discord server](https://discord.gg/pipecat) to get help and learn about what others are building.

## How Dynamic Alerts Work

Plant Chat's dynamic alert system learns your plant's normal environmental patterns and alerts when conditions change:

1. **Baseline Learning**: On startup, it fetches 24 hours of historical data to establish normal ranges
2. **Range Monitoring**: Alerts when readings fall outside the learned baseline ± configurable margins
3. **Shift Detection**: Notifies when readings deviate significantly from the baseline mean
4. **Smart Cooldowns**: Prevents alert spam with configurable cooldown periods
5. **Contextual Messages**: Provides plant-friendly explanations of what's happening

The system adapts to your specific environment, making it more intelligent than fixed threshold alerts.
