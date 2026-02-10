
import sys
import os
from pathlib import Path
import logging

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    print("Testing imports...")
    try:
        import stock_analysis_complete
        print("[OK] stock_analysis_complete imported successfully")
        
        # Check if heavy libraries are NOT imported yet
        import sys
        modules = sys.modules.keys()
        heavy_libs = ['torch', 'lightgbm', 'xgboost', 'pandas_ta']
        
        for lib in heavy_libs:
            if lib in modules:
                print(f"[WARNING] {lib} is already imported! Lazy loading might not be working fully.")
            else:
                print(f"[OK] {lib} is NOT imported yet (Lazy loading working)")
                
    except ImportError as e:
        print(f"[FAIL] Failed to import stock_analysis_complete: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Error during import test: {e}")
        return False
        
    return True

def test_classes():
    print("\nTesting classes...")
    try:
        from stock_analysis_complete import FeatureEngineer, StockPricePredictor
        
        fe = FeatureEngineer()
        print("[OK] FeatureEngineer instantiated")
        
        spp = StockPricePredictor()
        print("[OK] StockPricePredictor instantiated")
        
    except Exception as e:
        print(f"[FAIL] Error instantiating classes: {e}")
        return False
        
    return True

def test_dqn_import():
    print("\nTesting DQN import...")
    try:
        from core.dqn_agent import DQNTradingAgent
        print("[OK] DQNTradingAgent imported from core.dqn_agent")
        
        agent = DQNTradingAgent(n_features=10)
        print("[OK] DQNTradingAgent instantiated")
        
    except ImportError as e:
        print(f"[FAIL] Failed to import DQNTradingAgent: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Error testing DQN agent: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = True
    if not test_imports(): success = False
    if not test_classes(): success = False
    if not test_dqn_import(): success = False
    
    if success:
        print("\n[SUCCESS] All verification tests passed!")
        sys.exit(0)
    else:
        print("\n[FAIL] Verification tests failed!")
        sys.exit(1)
