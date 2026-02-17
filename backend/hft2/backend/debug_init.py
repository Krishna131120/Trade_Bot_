
import sys
import os
import logging

# Add current directory to path
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

try:
    print("Attempting to import testindia...")
    from testindia import StockTradingBot
    print("SUCCESS: StockTradingBot imported from testindia")
    
    print("Attempting to instantiate StockTradingBot with empty tickers...")
    config = {
        "tickers": [],  # Empty list as in web_backend.py default
        "mode": "paper",
        "riskLevel": "MEDIUM",
        "stop_loss_pct": 0.05,
        "max_capital_per_trade": 0.25,
        "max_trade_limit": 150,
        "starting_balance": 10000,
        "capital": 10000,
        "margin": 0,
        "target_profit_pct": 0.1,
        "max_drawdown_pct": 0.1,
        "use_risk_reward": True,
        "risk_reward_ratio": 2.0
    }
    bot = StockTradingBot(config)
    print("SUCCESS: StockTradingBot instantiated")
    
except ImportError as e:
    print(f"FAILURE: ImportError: {e}")
except Exception as e:
    print(f"FAILURE: Exception during initialization: {e}")
    import traceback
    traceback.print_exc()
