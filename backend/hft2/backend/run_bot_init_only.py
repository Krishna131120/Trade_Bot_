#!/usr/bin/env python3
"""
Run ONLY bot initialization in the main thread (no server, no dashboard).
Use this to see in the terminal exactly where init hangs or if it completes.

Usage: from backend/hft2/backend run:  python run_bot_init_only.py
"""
import os
import sys

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
        print(f"[run_bot_init_only] Loaded env from {ENV_FILE}")
    except ImportError:
        pass

def main():
    print("[run_bot_init_only] Importing web_backend (this may take a moment)...")
    sys.stdout.flush()
    # Import after path/env so we use same env
    import web_backend
    print("[run_bot_init_only] Calling initialize_bot() in main thread...")
    sys.stdout.flush()
    bot = web_backend.initialize_bot()
    if bot:
        print(f"[run_bot_init_only] OK - Bot initialized: {type(bot).__name__}")
    else:
        print("[run_bot_init_only] Bot init returned None")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
