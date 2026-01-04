# NEM Price Forecasting

**Live Dashboard:** [nem-price-forecasting.vercel.app](https://nem-price-forecasting.vercel.app/)

## Problem Statement

Australia's National Electricity Market (NEM) experiences extreme price volatility, with prices swinging from -$1,000 to +$16,600 per MWh within hours. Battery arbitrage can profit by charging during price troughs and discharging during spikes, but which algorithm maximizes returns?

This project benchmarks multiple trading strategies using real AEMO dispatch data to find the optimal approach for battery arbitrage.

---

## What I Built

A full-stack simulation engine that:
- Downloads and processes real 5-minute dispatch price data from AEMO NEMWEB
- Models battery physics with realistic constraints (capacity, power limits, round-trip efficiency)
- Implements 5 trading strategies of varying complexity with algorithmic analysis
- Includes EMA-based price forecasting for predictive trading decisions
- Provides a real-time Next.js dashboard deployed on Vercel
- Automates data updates every 15 minutes via GitHub Actions

---

## Dataset

| Attribute | Details |
|-----------|---------|
| Source | [AEMO NEMWEB](https://nemweb.com.au/) |
| Granularity | 5-minute dispatch intervals |
| Regions | SA1, NSW1, VIC1, QLD1, TAS1 |
| Target Variable | Regional Reference Price (RRP) in $/MWh |
| Update Frequency | Every 15 minutes (automated) |

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Backend | Python 3.11+, Pandas, NumPy, Matplotlib |
| Frontend | Next.js 15, React, Recharts |
| Data Pipeline | BeautifulSoup4, Requests, GitHub Actions |
| Deployment | Vercel (dashboard), GitHub Pages (data) |
| Testing | pytest (unit + E2E tests) |

**Algorithmic Techniques:** Dynamic programming, sliding window, threshold-based trading, exponential moving average forecasting

---

## Methodology

### Trading Strategies Implemented

| Strategy | Complexity | Approach |
|----------|------------|----------|
| Perfect Foresight | O(n×m) | Dynamic programming upper bound using future knowledge |
| Dynamic Programming | O(n×m) | Optimal decisions with discrete SoC states |
| Greedy Threshold | O(n) | Buy below 25th percentile, sell above 75th percentile |
| Sliding Window | O(n×k) | Local min/max detection within configurable time window |
| Forecast (EMA) | O(n) | Predictive trading based on exponential moving average |

### Battery Model

The simulation uses a realistic battery model with configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Capacity | 100 MWh | Total energy storage capacity |
| Power Rating | 50 MW | Maximum charge/discharge rate |
| Efficiency | 90% | Round-trip efficiency (split as √0.9 on charge, √0.9 on discharge) |
| Interval | 5 minutes | Dispatch interval duration |

Efficiency losses are modeled symmetrically: charging 10 MWh draws 10.54 MWh from the grid, while discharging 10 MWh delivers 9.49 MWh to the grid.

---

## Results

**Test Configuration:** SA1 Region, 100 MWh / 50 MW Battery, 90% Efficiency

| Strategy | Profit | % of Optimal | Charge Cycles | Discharge Cycles |
|----------|--------|--------------|---------------|------------------|
| Perfect Foresight | $634,137 | 100.0% | 1,808 | 1,788 |
| Dynamic Programming | $634,137 | 100.0% | 1,808 | 1,788 |
| Forecast Ema | $156,812 | 24.7% | 1,451 | 1,430 |
| Sliding Window | $154,111 | 24.3% | 103 | 82 |
| Greedy | $128,100 | 20.2% | 226 | 203 |

**Key Insight:** The Perfect Foresight algorithm establishes a theoretical upper bound of $634,137. Real-world strategies without future knowledge achieve 20-24% of optimal, with the EMA-based forecast strategy performing best among implementable approaches.

*Results auto-update every 15 minutes via GitHub Actions.*

---

## Visualizations

![Price Distribution](charts/price_distribution.png)
![Strategy Comparison](charts/strategy_comparison.png)
![Battery Operation](charts/battery_operation_greedy.png)

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/nem-price-forecasting.git
cd nem-price-forecasting

# Install dependencies
pip install -r requirements.txt

# Download latest AEMO data
python download_aemo_data.py

# Run simulation with default settings
python main.py

# Run for specific region
python main.py --region SA1

# Custom battery parameters
python main.py --capacity 200 --power 100 --efficiency 0.85

# Run without chart generation
python main.py --no-charts
```

### Dashboard Local Development

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

---

## Project Structure

```
nem-price-forecasting/
├── data/                        # AEMO price data (auto-downloaded)
├── charts/                      # Generated visualizations
├── src/
│   ├── battery.py              # Battery physics model
│   ├── data_loader.py          # Data parsing and preprocessing
│   ├── forecasting.py          # EMA, Rolling Mean, Linear Trend predictors
│   ├── metrics.py              # Performance metrics (Sharpe ratio, drawdown)
│   ├── visualizer.py           # Chart generation
│   └── strategies/
│       ├── perfect_foresight.py    # DP optimal solution
│       ├── greedy.py               # Threshold-based trading
│       ├── sliding_window.py       # Local extrema detection
│       └── dynamic_programming.py  # DP wrapper
├── tests/
│   ├── test_strategies.py      # Unit tests for trading strategies
│   ├── test_modules.py         # Module import tests
│   └── test_e2e.py             # End-to-end pipeline tests
├── dashboard/                   # Next.js frontend application
│   ├── app/
│   │   ├── page.js             # Main dashboard component
│   │   └── api/                # REST API endpoints
│   └── public/                 # Static simulation results (JSON)
├── docs/
│   ├── API.md                  # REST API documentation
│   └── CONFIGURATION.md        # Configuration guide
├── .github/workflows/
│   ├── ci.yml                  # CI pipeline (lint, test, build)
│   └── update_data.yml         # Automated data refresh (every 15 min)
├── main.py                     # CLI entry point
├── download_aemo_data.py       # AEMO NEMWEB scraper
└── requirements.txt            # Python dependencies
```

---

## Automation

The project includes automated pipelines for continuous operation:

| Pipeline | Trigger | Actions |
|----------|---------|---------|
| CI | Push/PR to main | Lint (flake8), Type check (mypy), Unit tests, E2E tests |
| Data Update | Every 15 minutes | Fetch AEMO data, Run simulations, Update dashboard JSON, Commit results |
| Weekly Squash | Sundays at midnight | Compress auto-update commits to prevent repository bloat |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_strategies.py -v

# Run E2E smoke tests
python -m pytest tests/test_e2e.py -v --tb=short
```

---

## API Documentation

See [docs/API.md](docs/API.md) for complete REST API reference.

**Example Endpoints:**

```bash
# Get price data
GET /api/prices?region=SA1&hours=24

# Run simulation
POST /api/simulation
Content-Type: application/json
{"region": "SA1", "strategy": "greedy", "battery": {"capacity": 100, "power": 50}}
```

---

## Key Learnings

- **Granularity matters:** 5-minute dispatch data captures price spikes that hourly aggregates miss entirely
- **Efficiency compounds:** 90% round-trip efficiency requires an 11% price spread just to break even
- **Simple can outperform:** In highly volatile markets, simple threshold strategies avoid overfitting to noise
- **Benchmark against optimal:** Perfect Foresight DP establishes the theoretical ceiling for strategy comparison

---

## Limitations

- Historical backtesting only; past performance does not guarantee future results
- No slippage, bid-ask spread, or grid constraints modeled
- Assumes perfect execution at 5-minute settlement prices
- Weather and demand forecasts not yet integrated

---

## Future Improvements

- Integrate Bureau of Meteorology weather forecasts for demand prediction
- Extend to multi-region arbitrage optimization (cross-border flows)
- Implement real-time WebSocket price streaming
- Add walk-forward validation for strategy robustness testing

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

**Data Source:** [AEMO NEMWEB](https://nemweb.com.au/) | **Dashboard:** [nem-price-forecasting.vercel.app](https://nem-price-forecasting.vercel.app/)