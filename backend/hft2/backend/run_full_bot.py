#!/usr/bin/env python3
"""
Run the FULL live prediction system:
- Bot initialization
- Historical data fetching
- Feature engineering
- ML predictions & analysis
- Trading decisions & execution
- Real-time monitoring
- All HFT2/backend functions

Usage: from backend/hft2/backend run:  python run_full_bot.py
Press Ctrl+C to stop gracefully.
"""
import os
import sys
import signal

# Run from backend/hft2/backend
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Load env from backend/hft2/env
ENV_FILE = os.path.join(os.path.dirname(BACKEND_DIR), "env")
if os.path.exists(ENV_FILE):
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        print(f"[run_full_bot] Loaded env from {ENV_FILE}")
    except ImportError:
        pass

# Global bot reference for signal handler
bot_instance = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global bot_instance
    print("\n[run_full_bot] Stopping bot gracefully...")
    if bot_instance:
        try:
            bot_instance.stop()
            print("[run_full_bot] Bot stopped successfully")
        except Exception as e:
            print(f"[run_full_bot] Error stopping bot: {e}")
    sys.exit(0)

def main():
    global bot_instance
    signal.signal(signal.SIGINT, signal_handler)
    
    print("[run_full_bot] Importing web_backend (this may take a moment)...")
    sys.stdout.flush()
    import web_backend
    
    print("[run_full_bot] Initializing bot...")
    sys.stdout.flush()
    bot_instance = web_backend.initialize_bot()
    
    if not bot_instance:
        print("[run_full_bot] ERROR - Bot initialization failed!")
        return
    
    print(f"[run_full_bot] Bot initialized: {type(bot_instance).__name__}")
    print(f"[run_full_bot] Mode: {bot_instance.config.get('mode', 'unknown')}")
    print(f"[run_full_bot] Watchlist: {', '.join(bot_instance.config.get('tickers', []))}")
    print("[run_full_bot] Starting full trading system...")
    print("[run_full_bot] This will run:")
    print("  - Historical data fetching")
    print("  - Feature engineering")
    print("  - ML predictions & analysis")
    print("  - Trading decisions & execution")
    print("  - Real-time monitoring")
    print("[run_full_bot] Press Ctrl+C to stop")
    sys.stdout.flush()
    
    # Start the bot - this runs the full trading loop
    try:
        bot_instance.start()
    except KeyboardInterrupt:
        print("\n[run_full_bot] Interrupted by user")
    except Exception as e:
        print(f"[run_full_bot] Error running bot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bot_instance:
            try:
                bot_instance.stop()
            except:
                pass

if __name__ == "__main__":
    main()
