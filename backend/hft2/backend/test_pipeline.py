#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full HFT2 backend pipeline test.
Run: python test_pipeline.py > test_output.txt 2>&1
Tests: web_backend API endpoints, bot initialization, start-with-symbol, predictions.
"""
import os, sys, time, json, importlib.util, traceback

# --- force stdout to utf-8 ---
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(os.path.dirname(BACKEND_DIR), "env")

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
    print(f"[OK] Loaded env from {ENV_FILE}")
except Exception as e:
    print(f"[WARN] Could not load env: {e}")

import requests

WEB_BACKEND_URL = "http://127.0.0.1:5000"
TEST_TICKER = "TCS.NS"
PASS = "[PASS]"
FAIL = "[FAIL]"

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def chk(label, ok, detail=""):
    tag = PASS if ok else FAIL
    msg = f"  {tag} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return ok

# ─────────────────────────────────────────────────────────────
# STEP 1: web_backend connectivity
# ─────────────────────────────────────────────────────────────
sep("STEP 1: web_backend.py connectivity (port 5000)")
backend_running = False
try:
    r = requests.get(f"{WEB_BACKEND_URL}/api/health", timeout=5)
    backend_running = r.status_code == 200
    chk("GET /api/health", backend_running, r.text[:100])
except Exception as e:
    chk("GET /api/health", False, str(e))
    print("  INFO: web_backend not running. Start with:  python web_backend.py --port 5000")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# STEP 2: List routes
# ─────────────────────────────────────────────────────────────
sep("STEP 2: API routes")
try:
    r = requests.get(f"{WEB_BACKEND_URL}/openapi.json", timeout=5)
    routes = r.json().get("paths", {})
    for route in sorted(routes):
        methods = ",".join(m.upper() for m in routes[route].keys())
        print(f"  {methods:12s}  {route}")
except Exception as e:
    print(f"  {FAIL} Could not fetch routes: {e}")

# ─────────────────────────────────────────────────────────────
# STEP 3: Bot status before start
# ─────────────────────────────────────────────────────────────
sep("STEP 3: Bot status BEFORE /api/start")
try:
    r = requests.get(f"{WEB_BACKEND_URL}/api/status", timeout=5)
    chk("GET /api/status", r.status_code == 200, r.text[:200])
except Exception as e:
    chk("GET /api/status", False, str(e))

# ─────────────────────────────────────────────────────────────
# STEP 4: POST /api/start
# ─────────────────────────────────────────────────────────────
sep("STEP 4: POST /api/start  (initialize + start bot)")
try:
    print("  INFO: Calling /api/start (waits up to 60s for init)...")
    t0 = time.time()
    r = requests.post(f"{WEB_BACKEND_URL}/api/start", timeout=90)
    elapsed = time.time() - t0
    chk("POST /api/start", r.status_code == 200, f"({elapsed:.1f}s) {r.text[:200]}")
except Exception as e:
    chk("POST /api/start", False, str(e))

# ─────────────────────────────────────────────────────────────
# STEP 5: Bot status after start
# ─────────────────────────────────────────────────────────────
sep("STEP 5: Bot status AFTER /api/start")
is_running = False
try:
    r = requests.get(f"{WEB_BACKEND_URL}/api/status", timeout=5)
    d = r.json()
    is_running = d.get("is_running", False)
    chk("is_running == True", is_running, str(d))
except Exception as e:
    chk("GET /api/status", False, str(e))

# ─────────────────────────────────────────────────────────────
# STEP 6: Watchlist
# ─────────────────────────────────────────────────────────────
sep("STEP 6: Watchlist")
tickers = [TEST_TICKER]
try:
    r = requests.get(f"{WEB_BACKEND_URL}/api/watchlist", timeout=5)
    if r.status_code == 200:
        data = r.json()
        tickers = data if isinstance(data, list) else data.get("tickers", [TEST_TICKER])
    chk("GET /api/watchlist", r.status_code == 200, f"Tickers: {tickers}")
    if not tickers:
        print(f"  INFO: Watchlist empty, adding {TEST_TICKER}")
        ra = requests.post(f"{WEB_BACKEND_URL}/api/watchlist/add/{TEST_TICKER}", timeout=5)
        print(f"  Add: {ra.text[:100]}")
        tickers = [TEST_TICKER]
except Exception as e:
    chk("GET /api/watchlist", False, str(e))

ticker = tickers[0] if tickers else TEST_TICKER
print(f"\n  => Testing with ticker: {ticker}")

# ─────────────────────────────────────────────────────────────
# STEP 7: POST /api/bot/start-with-symbol
# ─────────────────────────────────────────────────────────────
sep(f"STEP 7: POST /api/bot/start-with-symbol  (HFT2 pipeline for {ticker})")
try:
    print(f"  INFO: Triggering full HFT2 analysis for {ticker}...")
    print(f"  INFO: analyze_stock -> predictions -> trading decision")
    t0 = time.time()
    r = requests.post(
        f"{WEB_BACKEND_URL}/api/bot/start-with-symbol",
        json={"symbol": ticker},
        timeout=30,
    )
    elapsed = time.time() - t0
    chk("POST /api/bot/start-with-symbol", r.status_code == 200, f"({elapsed:.1f}s)  {r.text[:500]}")
except Exception as e:
    chk("POST /api/bot/start-with-symbol", False, str(e))

# ─────────────────────────────────────────────────────────────
# STEP 8: GET /api/bot-data
# ─────────────────────────────────────────────────────────────
sep("STEP 8: GET /api/bot-data")
try:
    r = requests.get(f"{WEB_BACKEND_URL}/api/bot-data", timeout=10)
    chk("GET /api/bot-data", r.status_code == 200)
    d = r.json()
    for k, v in sorted(d.items()):
        if isinstance(v, bool):
            print(f"    {k}: {v}")
        elif isinstance(v, (int, float)):
            print(f"    {k}: {v}")
        elif isinstance(v, str):
            print(f"    {k}: {v[:80]}")
        elif isinstance(v, dict):
            print(f"    {k}: dict({len(v)} keys) -> {list(v.keys())[:6]}")
        elif isinstance(v, list):
            print(f"    {k}: list({len(v)} items)")
except Exception as e:
    chk("GET /api/bot-data", False, str(e))

# ─────────────────────────────────────────────────────────────
# STEP 9: Direct testindia.py StockTradingBot + analyze_stock
# ─────────────────────────────────────────────────────────────
sep("STEP 9: Direct testindia.py StockTradingBot + analyze_stock")
print(f"  INFO: Testing {ticker}  (3mo data, 7-day prediction)")
print(f"  INFO: First run downloads data + trains models (may take 5-15 min)...")

try:
    spec = importlib.util.spec_from_file_location("testindia",
                os.path.join(BACKEND_DIR, "testindia.py"))
    testindia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(testindia)
    chk("import testindia", True)
    chk("StockTradingBot class exists", hasattr(testindia, "StockTradingBot"))

    config = {
        "tickers": [ticker],
        "starting_balance": 10000,
        "mode": "paper",
        "capital": 10000,
        "margin": 0,
        "stop_loss_pct": 0.05,
        "max_capital_per_trade": 0.25,
        "max_trade_limit": 5,
        "max_drawdown_pct": 0.1,
        "target_profit_pct": 0.1,
        "use_risk_reward": True,
        "risk_reward_ratio": 2.0,
        "period": "3mo",
        "prediction_days": 7,
        "benchmark_tickers": ["^NSEI"],
        "sleep_interval": 30,
        "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
        "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
    }

    print("  INFO: Creating StockTradingBot (no FinBERT download now)...")
    t0 = time.time()
    bot = testindia.StockTradingBot(config)
    elapsed = time.time() - t0
    chk(f"StockTradingBot.__init__ in {elapsed:.1f}s (<30s expected)", elapsed < 30, f"{elapsed:.1f}s")
    chk("bot._advanced_sentiment_analyzer is None (lazy, not downloaded)", bot._advanced_sentiment_analyzer is None)
    chk("bot.stock_analyzer is Stock", isinstance(bot.stock_analyzer, testindia.Stock))

    print(f"\n  INFO: Running analyze_stock({ticker}) ... this may take a while ...")
    t0 = time.time()
    result = bot.stock_analyzer.analyze_stock(
        ticker,
        benchmark_tickers=["^NSEI"],
        prediction_days=7,
        training_period="3mo",
        bot_running=True,
    )
    elapsed = time.time() - t0
    ok = isinstance(result, dict) and result.get("success", False)
    chk(f"analyze_stock returned success in {elapsed:.1f}s", ok, f"keys: {list(result.keys())[:10]}" if isinstance(result, dict) else str(result)[:200])

    if ok:
        print(f"\n  === PREDICTION RESULTS for {ticker} ===")
        for k in ["ticker", "current_price", "predicted_price", "prediction_direction",
                  "ensemble_prediction", "recommendation", "confidence",
                  "buy_score", "sell_score", "trade_signal"]:
            if k in result:
                print(f"    {k}: {result[k]}")
        if "ml_predictions" in result:
            mp = result["ml_predictions"]
            print(f"    ml_predictions type: {type(mp)}")
            if isinstance(mp, dict):
                for mk, mv in list(mp.items())[:5]:
                    print(f"      {mk}: {str(mv)[:80]}")
        # Test make_trading_decision
        print(f"\n  INFO: Testing make_trading_decision with analysis result ...")
        try:
            decision = bot.make_trading_decision(result)
            chk("make_trading_decision", True, f"result: {str(decision)[:200]}")
        except Exception as de:
            chk("make_trading_decision", False, str(de))
    else:
        print(f"  Details: {str(result)[:500]}")

except Exception as e:
    chk("testindia.py pipeline", False, str(e))
    traceback.print_exc()

sep("TEST COMPLETE")
print("  PASS = step worked correctly")
print("  FAIL = step had an error")
print("  Check above for full prediction results")
print()
