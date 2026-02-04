# Stock Analysis & ML Prediction System

A comprehensive stock market analysis and prediction system that combines traditional machine learning models with deep reinforcement learning to provide intelligent trading predictions. The system supports both US and Indian (NSE) stock markets with automatic fallback mechanisms.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Technical Indicators](#technical-indicators)
- [Machine Learning Models](#machine-learning-models)
- [Usage](#usage)
- [API Interface](#api-interface)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## 🎯 Overview

This system provides end-to-end stock analysis and prediction capabilities:

- **Data Ingestion**: Fetches historical price data, financial statements, and news from Yahoo Finance with automatic fallback to NSE Bhav Copy for Indian stocks
- **Feature Engineering**: Calculates 50+ technical indicators automatically
- **ML Models**: Ensemble of 3 traditional ML models (Random Forest, LightGBM, XGBoost) for price prediction
- **RL Agent**: Deep Q-Network (DQN) for trading action recommendations
- **Intelligent Predictions**: Combines all models with overfitting detection and risk analysis
- **API Interface**: RESTful API and MCP-style adapter for integration

### Supported Markets

- **US Stocks**: AAPL, MSFT, TSLA, GOOGL, etc.
- **Indian NSE Stocks**: RPOWER.NS, TCS.NS, INFY.NS, RELIANCE.NS, etc.

---

## ✨ Features

### Data Management
- ✅ Automatic data fetching from Yahoo Finance (2 years historical data)
- ✅ Automatic fallback to NSE Bhav Copy for Indian stocks
- ✅ Data caching to reduce API calls
- ✅ Financial statements (Income, Balance Sheet, Cash Flow)
- ✅ News sentiment analysis
- ✅ Data format consistency across sources

### Technical Analysis
- ✅ **50+ Technical Indicators** calculated automatically:
  - Momentum: RSI, MACD, Stochastic, CCI, MFI, ROC, TRIX, Aroon, etc.
  - Trend: SMA (10/20/50/200), EMA, ADX, PSAR, Keltner Channels, Donchian
  - Volatility: Bollinger Bands, ATR, Historical Volatility
  - Volume: OBV, CMF, AD, VROC, EMV
  - Support/Resistance: Pivot Points, Fibonacci Retracements
  - Patterns: Doji, Hammer, Engulfing patterns (if TA-Lib available)
  - Advanced: VWAP, Sharpe Ratio, Price-to-MA ratios

### Machine Learning
- ✅ **3 ML Models** for price prediction:
  - Random Forest Regressor
  - LightGBM Regressor
  - XGBoost Regressor
- ✅ **Ensemble Prediction** with weighted averaging
- ✅ **Adaptive Model Complexity** based on data size
- ✅ **Feature Selection** to prevent overfitting
- ✅ **Multiple Horizons**: Intraday (1 day), Short (5 days), Long (30 days)

### Reinforcement Learning
- ✅ **DQN Agent** for trading action recommendations
- ✅ Actions: LONG, SHORT, HOLD
- ✅ Experience replay buffer
- ✅ Epsilon-greedy exploration
- ✅ Reward-based learning

### Prediction Features
- ✅ **Unified Predictions** combining ML and RL models
- ✅ **Confidence Scores** for each prediction
- ✅ **Overfitting Detection** with warnings
- ✅ **Risk Analysis** (volatility, expected moves)
- ✅ **Detailed Reasoning** with technical indicators and news sentiment
- ✅ **Realistic Prediction Validation**

### API & Integration
- ✅ RESTful API server (FastAPI)
- ✅ MCP-style adapter for orchestrator integration
- ✅ Comprehensive request/response logging
- ✅ JSON serialization handling
- ✅ Error handling and validation

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
# Navigate to project directory
cd /path/to/project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import yfinance, pandas, numpy, torch, lightgbm, xgboost; print('All dependencies installed!')"
```

### Optional: TA-Lib for Advanced Pattern Recognition

For enhanced candlestick pattern recognition, install TA-Lib:

**Windows:**
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.28-cp39-cp39-win_amd64.whl
```

**Linux/Mac:**
```bash
# Install TA-Lib C library first
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Then install Python wrapper
pip install TA-Lib
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Adapter / API Server                  │
│              (core/mcp_adapter.py, api_server.py)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Stock Analysis Complete System                  │
│              (stock_analysis_complete.py)                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ EnhancedData     │  │ Feature          │                 │
│  │ Ingester         │  │ Engineer         │                 │
│  │                  │  │                  │                 │
│  │ - yfinance       │  │ - 50+ Indicators│                 │
│  │ - NSE Bhav       │  │ - Feature        │                 │
│  │ - Financials     │  │   Selection      │                 │
│  │ - News           │  │                  │                 │
│  └────────┬─────────┘  └────────┬─────────┘                 │
│           │                     │                            │
│           └──────────┬──────────┘                            │
│                      │                                       │
│  ┌───────────────────▼───────────────────┐                  │
│  │      Stock Price Predictor            │                  │
│  │  (Random Forest + LightGBM + XGBoost) │                  │
│  └───────────────────┬───────────────────┘                  │
│                      │                                       │
│  ┌───────────────────▼───────────────────┐                  │
│  │      DQN Trading Agent               │                  │
│  │  (Deep Reinforcement Learning)        │                  │
│  └───────────────────┬───────────────────┘                  │
│                      │                                       │
│  ┌───────────────────▼───────────────────┐                  │
│  │      Prediction Engine                │                  │
│  │  (Ensemble + Risk Analysis)           │                  │
│  └───────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Data Storage                             │
│  - data/cache/        (Price data, financials)              │
│  - data/features/     (Technical indicators)                │
│  - models/            (Trained ML/RL models)                │
│  - data/logs/         (Request logs, predictions)           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Data Ingestion**: Fetch from yfinance → Fallback to NSE Bhav if needed
2. **Feature Engineering**: Calculate 50+ technical indicators
3. **Model Training**: Train 3 ML models + DQN agent (if not cached)
4. **Prediction**: Generate predictions from all models
5. **Ensemble**: Combine predictions with weighted averaging
6. **Analysis**: Add overfitting detection, risk analysis, reasoning

---

## 📊 Data Sources

### Yahoo Finance (Primary)

- **Source**: `yfinance` library
- **Data**: Price history (OHLCV), financial statements, news
- **Period**: 2 years by default
- **Format**: Standardized DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits

### NSE Bhav Copy (Fallback for Indian Stocks)

- **Source**: NSE India official website
- **Trigger**: Automatic fallback when yfinance fails for `.NS` symbols
- **Data**: Price history (OHLCV) only
- **Format**: Converted to match yfinance format exactly
- **Limitations**: 
  - No news data (handled gracefully in output)
  - No dividends/stock splits (set to 0.0)
  - Slower than yfinance (day-by-day download)

### Data Format Consistency

Both sources produce identical DataFrame format:
```python
Columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
Index: DatetimeIndex (timezone-naive)
```

**Note**: Dividends and Stock Splits are excluded from feature calculation, so the difference (0.0 vs actual values) doesn't affect predictions.

---

## 📈 Technical Indicators

The system calculates **50+ technical indicators** organized into 7 categories:

### 1. Momentum Indicators (14)
- **RSI_14**: Relative Strength Index (14-period)
- **MACD**: Moving Average Convergence Divergence (12, 26, 9)
- **MACD_signal**: MACD signal line
- **MACD_hist**: MACD histogram
- **STOCH_k/d**: Stochastic Oscillator (14, 3, 3)
- **WILLR**: Williams %R (14-period)
- **CCI**: Commodity Channel Index (20-period)
- **MFI**: Money Flow Index (14-period)
- **ROC**: Rate of Change (12-period)
- **TRIX**: Triple Exponential Average (15-period)
- **CMO**: Chande Momentum Oscillator (14-period)
- **AROON_up/down/osc**: Aroon Indicator (25-period)
- **UO**: Ultimate Oscillator

### 2. Trend Indicators (11+)
- **SMA_10/20/50/200**: Simple Moving Averages
- **EMA_12/26**: Exponential Moving Averages
- **ADX**: Average Directional Index (14-period)
- **DMP/DMN**: Directional Movement indicators
- **PSAR**: Parabolic SAR
- **KC_upper/middle/lower**: Keltner Channels (20, 2)
- **DC_upper/middle/lower**: Donchian Channels (20-period)
- **MA_alignment**: Moving average alignment score (0-1)
- **trend_direction**: Trend direction (-1, 0, 1)

### 3. Volatility Indicators (5+)
- **BB_lower/middle/upper**: Bollinger Bands (20, 2 std)
- **BB_width**: Bollinger Band width
- **BB_pct**: Bollinger Band %B
- **ATR**: Average True Range (14-period)
- **STD_20**: 20-period standard deviation
- **volatility_20**: 20-period annualized volatility

### 4. Volume Indicators (7+)
- **OBV**: On-Balance Volume
- **Volume_SMA_20**: 20-period volume moving average
- **AD**: Accumulation/Distribution Line
- **CMF**: Chaikin Money Flow (20-period)
- **VROC**: Volume Rate of Change (12-period)
- **EMV**: Ease of Movement (14-period)
- **volume_ratio**: Volume to volume MA ratio
- **volume_trend**: Volume trend direction

### 5. Support/Resistance (5+)
- **pivot**: Pivot point
- **pivot_r1/r2**: Resistance levels
- **pivot_s1/s2**: Support levels
- **fib_0.236/0.382/0.500/0.618**: Fibonacci retracement levels
- **price_position**: Price position in 50-day range (0-1)

### 6. Pattern Features (5+)
- **CDL_DOJI**: Doji candlestick pattern (if TA-Lib available)
- **CDL_HAMMER**: Hammer candlestick pattern (if TA-Lib available)
- **higher_high**: Higher high pattern
- **lower_low**: Lower low pattern
- **gap_up/gap_down**: Gap patterns
- **bullish_engulf/bearish_engulf**: Engulfing patterns

### 7. Advanced Analytics (8+)
- **price_to_sma_10/50/200**: Price-to-MA ratios
- **rsi_divergence**: RSI divergence detection
- **macd_hist_increasing**: MACD histogram trend
- **bb_position**: Bollinger Band position (0-1)
- **VWAP**: Volume Weighted Average Price
- **daily_return**: Daily return percentage
- **daily_return_ma_5**: 5-day return moving average
- **sharpe_20**: 20-period Sharpe ratio
- **returns_autocorr**: Returns autocorrelation

### Feature Selection

The system uses **adaptive feature selection** to prevent overfitting:
- **Rule**: ~10 samples per feature
- **Priority**: Features with higher correlation to target
- **Minimum**: At least 20 features for small datasets
- **Maximum**: All features for large datasets (>500 samples)

---

## 🤖 Machine Learning Models

### Price Prediction Models

#### 1. Random Forest Regressor
- **Type**: Ensemble of decision trees
- **Features**: Adaptive based on data size
- **Hyperparameters** (adaptive):
  - `n_estimators`: 50-200 (based on data size)
  - `max_depth`: 5-15 (deeper for more data)
  - `min_samples_split`: 5-20 (higher for less data)
  - `min_samples_leaf`: 2-10 (higher for less data)

#### 2. LightGBM Regressor
- **Type**: Gradient boosting framework
- **Features**: Adaptive feature selection
- **Hyperparameters** (adaptive):
  - `n_estimators`: 50-200
  - `learning_rate`: 0.01-0.1 (lower for more data)
  - `num_leaves`: 15-127 (more for more data)
  - `min_child_samples`: 5-20 (higher for less data)
  - `reg_alpha/reg_lambda`: 0.1-1.0 (higher for less data)

#### 3. XGBoost Regressor
- **Type**: Extreme gradient boosting
- **Features**: Adaptive feature selection
- **Hyperparameters** (adaptive):
  - `n_estimators`: 50-200
  - `learning_rate`: 0.01-0.1
  - `max_depth`: 3-10
  - `min_child_weight`: 1-5
  - `reg_alpha/reg_lambda`: 0.1-1.0
  - `gamma`: 0-0.5

### Ensemble Prediction

Predictions are combined using **weighted averaging**:
- **Weight**: Based on test R² score
- **Exclusion**: Models with R² < -0.5 are excluded
- **Formula**: `weight = max(0, R²_score) / sum(max(0, all_R²_scores))`

### Reinforcement Learning: DQN Agent

#### Architecture
- **Network**: 3-layer fully connected neural network
- **Layers**: [256, 128, 64] neurons (default)
- **Activation**: ReLU (hidden), Linear (output)
- **Input**: Feature vector (normalized)
- **Output**: Q-values for 3 actions (LONG, SHORT, HOLD)

#### Training
- **Episodes**: Adaptive (based on data size)
- **Replay Buffer**: 10,000 experiences
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Gamma (discount)**: 0.99
- **Epsilon**: Decays from 1.0 to 0.01
- **Reward Function**: Based on actual returns

#### Actions
- **LONG (0)**: Buy/Long position
- **SHORT (1)**: Sell/Short position
- **HOLD (2)**: Hold current position

### Overfitting Prevention

The system implements multiple strategies to prevent overfitting:

1. **Adaptive Feature Selection**: Reduces features for small datasets
2. **Adaptive Model Complexity**: Adjusts hyperparameters based on data size
3. **Weighted Ensemble**: Down-weights poorly performing models
4. **Stronger Regularization**: Higher L1/L2 penalties for small datasets
5. **Overfitting Detection**: Warnings for high train-test gap, high prediction spread, unrealistic predictions

---

## 💻 Usage

### Command Line Interface

#### Interactive Menu

```bash
python stock_analysis_complete.py
```

Menu options:
1. **Fetch Stock Data**: Download all data from Yahoo Finance
2. **Calculate Technical Indicators**: Calculate 50+ indicators
3. **View Financial Statements**: View previously downloaded data
4. **View Technical Indicators**: View previously calculated indicators
5. **Complete Analysis**: Fetch data + Calculate indicators + Show everything
6. **Train ALL 4 Models**: Train RF + LightGBM + XGBoost + DQN
7. **Predict with Ensemble**: Unified prediction from all 4 models
8. **Exit**

#### Direct Function Calls

```python
from stock_analysis_complete import (
    EnhancedDataIngester,
    FeatureEngineer,
    train_ml_models,
    predict_stock_price
)

# Initialize
ingester = EnhancedDataIngester()
engineer = FeatureEngineer()

# Fetch data
all_data = ingester.fetch_all_data("RPOWER.NS", period="2y")
df = all_data['price_history']

# Calculate features
features_df = engineer.calculate_all_features(df, "RPOWER.NS")
engineer.save_features(features_df, "RPOWER.NS")

# Train models
train_ml_models("RPOWER.NS", horizon="intraday", verbose=True)

# Predict
prediction = predict_stock_price("RPOWER.NS", horizon="intraday", verbose=True)
print(f"Action: {prediction['action']}")
print(f"Confidence: {prediction['confidence']}")
```

### NSE Bhav Copy (Direct Usage)

To specifically fetch from NSE Bhav Copy:

```python
from stock_analysis_complete import EnhancedDataIngester
from datetime import datetime, timedelta

ingester = EnhancedDataIngester()

# Fetch 2 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years

df = ingester.fetch_nse_bhav_historical("RPOWER", start_date, end_date)
print(f"Fetched {len(df)} rows from NSE Bhav Copy")
```

---

## 🔌 API Interface

### MCP Adapter

The MCP adapter provides tool-style interfaces for orchestrator integration:

```python
from core.mcp_adapter import MCPAdapter

adapter = MCPAdapter()

# Predict
result = adapter.predict(
    symbols=["RPOWER.NS", "TCS.NS"],
    horizon="intraday",
    risk_profile="moderate"
)

# Fetch data
result = adapter.fetch_data(
    symbols=["RPOWER.NS"],
    period="2y",
    include_features=True
)

# Train models
result = adapter.train_ml(
    symbols=["RPOWER.NS"],
    horizon="intraday",
    force_retrain=False
)

# Train RL
result = adapter.train_rl(
    symbol="RPOWER.NS",
    horizon="intraday",
    n_episodes=100
)
```

### RESTful API Server

Start the API server:

```bash
python api_server.py
```

The server runs on `http://localhost:8000` by default.

#### Endpoints

- `POST /predict`: Get predictions for symbols
- `POST /fetch_data`: Fetch and cache data
- `POST /train_ml`: Train ML models
- `POST /train_rl`: Train RL agent
- `GET /health`: Health check

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["RPOWER.NS"],
    "horizon": "intraday"
  }'
```

---

## ⚙️ Configuration

### Directory Structure

```
project/
├── data/
│   ├── cache/           # Price data, financials (JSON)
│   ├── features/        # Technical indicators (JSON)
│   └── logs/            # Request logs, predictions
│       └── mcp_requests/
├── models/              # Trained ML/RL models
├── core/
│   └── mcp_adapter.py   # MCP adapter
├── stock_analysis_complete.py  # Main system
└── api_server.py        # API server
```

### Key Configuration Variables

In `stock_analysis_complete.py`:

```python
# Directories
DATA_DIR = Path("data")
DATA_CACHE_DIR = DATA_DIR / "cache"
FEATURE_CACHE_DIR = DATA_DIR / "features"
MODEL_DIR = Path("models")
LOGS_DIR = DATA_DIR / "logs"
NSE_BHAV_CACHE_DIR = DATA_CACHE_DIR / "nse_bhav"

# Historical data period
HISTORICAL_PERIOD = "2y"

# Random seed for reproducibility
RANDOM_SEED = 42
```

### Prediction Horizons

- **intraday**: 1 day prediction
- **short**: 5 days prediction
- **long**: 30 days prediction

### Risk Profiles

- **conservative**: Lower risk tolerance
- **moderate**: Balanced risk
- **aggressive**: Higher risk tolerance

---

## 📁 File Structure

### Cache Files

- `data/cache/{SYMBOL}_all_data.json`: Price history, financials, news
- `data/cache/nse_bhav/{SYMBOL}_{DATE}.json`: NSE Bhav daily data
- `data/features/{SYMBOL}_features.json`: Latest technical indicators

### Model Files

- `models/{SYMBOL}_{HORIZON}_random_forest.pkl`: Random Forest model
- `models/{SYMBOL}_{HORIZON}_lightgbm.pkl`: LightGBM model
- `models/{SYMBOL}_{HORIZON}_xgboost.pkl`: XGBoost model
- `models/{SYMBOL}_{HORIZON}_features.pkl`: Feature columns used
- `models/{SYMBOL}_{HORIZON}_scaler.pkl`: Feature scaler
- `models/{SYMBOL}_{HORIZON}_dqn_agent.pt`: DQN agent checkpoint
- `models/{SYMBOL}_{HORIZON}_dqn_features.pkl`: DQN feature columns

### Log Files

- `data/logs/predictions.json`: All predictions (JSONL format)
- `data/logs/mcp_requests/{DATE}_requests.jsonl`: API requests
- `data/logs/mcp_requests/{DATE}_responses.jsonl`: API responses

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Insufficient data" Error

**Problem**: `Insufficient data for {SYMBOL}: X rows (need 100+)`

**Solution**:
- The system requires at least 50 rows (hard minimum)
- For best results, use 100+ rows
- Some indicators (like SMA_200) need more data but will work with NaN initially

#### 2. NSE Bhav Copy 403 Forbidden

**Problem**: `HTTP error downloading NSE data: 403 Client Error: Forbidden`

**Solution**:
- The system automatically retries with session refresh
- If persistent, reduce parallel downloads (already set to 5 workers)
- NSE may rate-limit; wait a few minutes and retry

#### 3. Overfitting Warnings

**Problem**: Warnings about high train-test gap or unrealistic predictions

**Solution**:
- The system automatically uses adaptive feature selection and model complexity
- For very small datasets (<100 rows), predictions may be less reliable
- Consider using a longer historical period (e.g., 5 years instead of 2)

#### 4. JSON Serialization Error

**Problem**: `Object of type bool is not JSON serializable`

**Solution**:
- Fixed in `mcp_adapter.py` with `_convert_to_json_serializable()`
- If you see this, ensure you're using the latest version

#### 5. Model Training Takes Too Long

**Problem**: Training takes 60-90 seconds per symbol

**Solution**:
- This is normal for the first training
- Models are cached; subsequent predictions are instant
- Use `force_retrain=False` to skip retraining if models exist

#### 6. NSE Bhav Copy is Slow

**Problem**: Fetching 2 years of data takes several minutes

**Solution**:
- NSE Bhav downloads day-by-day (limitation of NSE website)
- The system uses parallel downloads (5 workers) and connection pooling
- Data is cached; subsequent fetches are instant
- Consider using yfinance for faster initial fetch (if available)

#### 7. News Not Available for NSE Bhav

**Problem**: "BHAV does not support news" in prediction reason

**Solution**:
- This is expected; NSE Bhav Copy doesn't provide news data
- Technical analysis is still included in the reason
- Use yfinance if news is critical (when available)

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📝 Examples

### Example 1: Complete Analysis Workflow

```python
from stock_analysis_complete import (
    EnhancedDataIngester,
    FeatureEngineer,
    train_ml_models,
    predict_stock_price
)

symbol = "RPOWER.NS"

# Step 1: Fetch data
ingester = EnhancedDataIngester()
all_data = ingester.fetch_all_data(symbol, period="2y")
print(f"Data source: {all_data.get('price_history_metadata', {}).get('data_source', 'yfinance')}")

# Step 2: Calculate features
engineer = FeatureEngineer()
df = all_data['price_history']
features_df = engineer.calculate_all_features(df, symbol)
engineer.save_features(features_df, symbol)
print(f"Calculated {len(features_df.columns)} features")

# Step 3: Train models
train_result = train_ml_models(symbol, horizon="intraday", verbose=True)
print(f"Training success: {train_result.get('success', False)}")

# Step 4: Predict
prediction = predict_stock_price(symbol, horizon="intraday", verbose=True)
print(f"\nPrediction:")
print(f"  Action: {prediction['action']}")
print(f"  Confidence: {prediction['confidence']:.4f}")
print(f"  Expected Return: {prediction['expected_return']:.2%}")
print(f"  Reason: {prediction['reason'][:200]}...")
```

### Example 2: Batch Prediction

```python
from core.mcp_adapter import MCPAdapter

adapter = MCPAdapter()

symbols = ["RPOWER.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS"]
result = adapter.predict(
    symbols=symbols,
    horizon="short",  # 5-day prediction
    risk_profile="moderate"
)

for pred in result['predictions']:
    print(f"{pred['symbol']}: {pred['action']} (confidence: {pred['confidence']:.4f})")
```

### Example 3: Force NSE Bhav Copy

```python
from stock_analysis_complete import EnhancedDataIngester
from datetime import datetime, timedelta

ingester = EnhancedDataIngester()

# Force NSE Bhav by using fetch_nse_bhav_historical directly
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years

df = ingester.fetch_nse_bhav_historical("RPOWER", start_date, end_date)
print(f"Fetched {len(df)} rows from NSE Bhav Copy")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
```

### Example 4: View Technical Indicators

```python
from stock_analysis_complete import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.load_features("RPOWER.NS")

print("Latest Technical Indicators:")
print(f"  RSI_14: {features.get('RSI_14', 'N/A'):.2f}")
print(f"  MACD: {features.get('MACD', 'N/A'):.4f}")
print(f"  SMA_50: {features.get('SMA_50', 'N/A'):.2f}")
print(f"  BB_upper: {features.get('BB_upper', 'N/A'):.2f}")
print(f"  BB_lower: {features.get('BB_lower', 'N/A'):.2f}")
print(f"  ADX: {features.get('ADX', 'N/A'):.2f}")
```

---

## 📚 Additional Notes

### Model Isolation

Each stock symbol has **separate models**. Training one symbol does not affect others:
- Models are saved per symbol: `{SYMBOL}_{HORIZON}_*.pkl`
- Features are calculated per symbol: `{SYMBOL}_features.json`
- Data is cached per symbol: `{SYMBOL}_all_data.json`

### Data Freshness

- **Price Data**: Fetched fresh if cache is >1 day old
- **Features**: Recalculated if price data is updated
- **Models**: Cached until `force_retrain=True`

### Performance

- **First Prediction**: 60-90 seconds (includes training)
- **Subsequent Predictions**: 2-5 seconds (uses cached models)
- **Data Fetching**: 5-10 seconds (yfinance) or 2-5 minutes (NSE Bhav, first time)
- **Feature Calculation**: 1-3 seconds for 2 years of data

### Best Practices

1. **Use appropriate horizon**: Intraday for day trading, Short for swing trading, Long for position trading
2. **Monitor overfitting warnings**: High warnings indicate less reliable predictions
3. **Check data source**: NSE Bhav doesn't have news; yfinance is preferred when available
4. **Retrain periodically**: Models should be retrained monthly or when market conditions change significantly
5. **Use ensemble predictions**: Individual model predictions may vary; ensemble is more stable

---

## 📄 License

This project is for educational and research purposes. Use at your own risk. Stock market predictions are not guaranteed and should not be the sole basis for trading decisions.

---

## 🤝 Contributing

This is a personal project. For issues or suggestions, please create an issue or contact the maintainer.

---

## 📞 Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the code comments in `stock_analysis_complete.py`
3. Check the logs in `data/logs/`

---

**Last Updated**: November 2025

