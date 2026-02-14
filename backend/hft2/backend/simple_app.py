#!/usr/bin/env python3
"""
HFT2 backend - run on port 5001. Uses Dhan API for live portfolio when env (backend/hft2/env) has DHAN_ACCESS_TOKEN.
Same API as main backend /api/* so proxy from 8000 works.
"""
import logging
from typing import List, Optional
from datetime import datetime, timedelta
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

try:
    from dhan_client import get_live_portfolio, get_live_trades, get_dhan_token, place_order_market, fetch_holdings, fetch_positions, fetch_fund_limit
except ImportError:
    get_live_portfolio = get_live_trades = get_dhan_token = place_order_market = fetch_holdings = fetch_positions = fetch_fund_limit = None  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HFT2 API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
bot_state = {
    "isRunning": False,
    "config": {
        "mode": "paper",
        "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "riskLevel": "MEDIUM",
        "maxAllocation": 0.25,
        "stopLoss": 0.05,
    },
    "portfolio": {
        "totalValue": 1050000,
        "cash": 850000,
        "startingBalance": 1000000,
        "holdings": {
            "RELIANCE.NS": {"symbol": "RELIANCE.NS", "quantity": 50, "avgPrice": 2500, "currentPrice": 2600, "lastAction": "BUY"},
            "TCS.NS": {"symbol": "TCS.NS", "quantity": 100, "avgPrice": 3200, "currentPrice": 3300, "lastAction": "BUY"},
        },
        "tradeLog": [
            {"timestamp": (datetime.now() - timedelta(hours=2)).isoformat(), "symbol": "RELIANCE.NS", "action": "BUY", "quantity": 50, "price": 2500, "total": 125000},
            {"timestamp": (datetime.now() - timedelta(hours=1)).isoformat(), "symbol": "TCS.NS", "action": "BUY", "quantity": 100, "price": 3200, "total": 320000},
        ],
    },
    "chatMessages": [],
}


class ChatMessage(BaseModel):
    message: str


class BotConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    mode: str
    risk_level: str = Field(alias="riskLevel")
    max_allocation: float = Field(alias="maxAllocation")
    stop_loss: Optional[float] = Field(None, alias="stopLoss")


class WatchlistBulkBody(BaseModel):
    tickers: List[str]
    action: str = "ADD"


class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: int
    order_type: str = "MARKET"
    price: Optional[float] = None


@app.get("/")
async def root():
    return {"message": "HFT2 API - Running", "status": "operational"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/status")
async def status():
    mode = bot_state.get("config", {}).get("mode", "paper")
    dhan_token = get_dhan_token() if get_dhan_token else None
    return {
        "status": "healthy",
        "isRunning": bot_state["isRunning"],
        "mode": mode,
        "dhan_connected": bool(dhan_token),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/debug/dhan")
async def debug_dhan():
    """Debug endpoint to test Dhan connection"""
    from dhan_client import get_dhan_client_id
    token = get_dhan_token() if get_dhan_token else None
    client_id = get_dhan_client_id() if get_dhan_client_id else None
    if not token:
        return {"error": "No Dhan token found", "token": False, "client_id": bool(client_id)}
    
    if not fetch_fund_limit or not fetch_holdings or not fetch_positions:
        return {"error": "Dhan client functions not imported", "token": True}
    
    try:
        fund = fetch_fund_limit(token)
        holdings = fetch_holdings(token)
        positions = fetch_positions(token)
        return {
            "token": True,
            "client_id": bool(client_id),
            "fund": bool(fund),
            "holdings_count": len(holdings),
            "positions_count": len(positions),
            "holdings": holdings[:3] if holdings else [],  # First 3 for debugging
            "positions": positions[:3] if positions else [],
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc(), "token": True}


def _empty_portfolio():
    """No stale data in paper mode."""
    return {
        "totalValue": 0,
        "cash": 0,
        "startingBalance": 0,
        "holdings": {},
        "tradeLog": [],
    }


def _get_portfolio():
    """Live mode: real prices from Dhan. Paper mode: no stale data (empty portfolio)."""
    mode = bot_state.get("config", {}).get("mode", "paper")
    logger.info(f"[_get_portfolio] Mode: {mode}")
    if mode == "live":
        token = get_dhan_token() if get_dhan_token else None
        logger.info(f"[_get_portfolio] Dhan token available: {bool(token)}")
        if get_live_portfolio and token:
            try:
                live = get_live_portfolio()
                logger.info(f"[_get_portfolio] Dhan returned portfolio: {live is not None}, holdings: {len(live.get('holdings', {})) if live else 0}")
                if live is not None:
                    return live
            except Exception as e:
                logger.error(f"[_get_portfolio] Error fetching Dhan portfolio: {e}")
        logger.warning("[_get_portfolio] Live mode but no Dhan data - returning empty")
        return _empty_portfolio()
    # Paper: never return stub data
    logger.info("[_get_portfolio] Paper mode - returning empty portfolio")
    return _empty_portfolio()


@app.get("/api/bot-data")
async def get_bot_data():
    portfolio = _get_portfolio()
    return {
        **bot_state,
        "isRunning": bot_state["isRunning"],
        "portfolio": portfolio,
    }


@app.get("/api/portfolio")
async def get_portfolio():
    return _get_portfolio()


@app.get("/api/trades")
async def get_trades(limit: int = 10):
    if get_live_trades and get_dhan_token and get_dhan_token():
        trades = get_live_trades(limit)
        if trades is not None:
            return trades[:limit]
    return bot_state["portfolio"]["tradeLog"][:limit]


@app.get("/api/watchlist")
async def get_watchlist():
    return {"tickers": bot_state["config"]["tickers"]}


@app.post("/api/watchlist/add/{ticker}")
async def add_watchlist(ticker: str):
    t = ticker.upper().strip()
    if t not in bot_state["config"]["tickers"]:
        bot_state["config"]["tickers"].append(t)
    return {"status": "success", "message": f"Added {t}", "tickers": bot_state["config"]["tickers"]}


@app.delete("/api/watchlist/remove/{ticker}")
async def remove_watchlist(ticker: str):
    t = ticker.upper().strip()
    if t in bot_state["config"]["tickers"]:
        bot_state["config"]["tickers"].remove(t)
    return {"status": "success", "message": f"Removed {t}", "tickers": bot_state["config"]["tickers"]}


@app.post("/api/watchlist/bulk")
async def watchlist_bulk(body: WatchlistBulkBody):
    for t in (x.upper().strip() for x in body.tickers):
        if body.action.upper() == "ADD" and t not in bot_state["config"]["tickers"]:
            bot_state["config"]["tickers"].append(t)
        elif body.action.upper() == "REMOVE" and t in bot_state["config"]["tickers"]:
            bot_state["config"]["tickers"].remove(t)
    return {"status": "success", "tickers": bot_state["config"]["tickers"]}


@app.post("/api/bot/start")
async def start_bot():
    bot_state["isRunning"] = True
    return {"status": "success", "message": "Bot started", "isRunning": True}


@app.post("/api/bot/stop")
async def stop_bot():
    bot_state["isRunning"] = False
    return {"status": "success", "message": "Bot stopped", "isRunning": False}


@app.get("/api/settings")
async def get_settings():
    return bot_state["config"]


@app.post("/api/settings")
async def update_settings(config: BotConfig):
    bot_state["config"]["mode"] = config.mode
    bot_state["config"]["riskLevel"] = config.risk_level
    bot_state["config"]["maxAllocation"] = config.max_allocation
    if config.stop_loss is not None:
        bot_state["config"]["stopLoss"] = config.stop_loss
    return {"status": "success", "message": "Settings updated"}


@app.post("/api/chat")
async def chat(message: ChatMessage):
    bot_state["chatMessages"].append({"role": "user", "content": message.message, "timestamp": datetime.now().isoformat()})
    resp = "I'm the HFT trading assistant. How can I help you today?"
    bot_state["chatMessages"].append({"role": "assistant", "content": resp, "timestamp": datetime.now().isoformat()})
    return {"response": resp, "messages": bot_state["chatMessages"]}


@app.get("/api/live-status")
async def live_status():
    mode = bot_state["config"]["mode"]
    # When Live Trading is selected and Dhan token is set, show as connected (real data)
    connected = mode == "live" and bool(get_dhan_token and get_dhan_token())
    return {"connected": connected, "mode": mode, "lastUpdate": datetime.now().isoformat()}


@app.post("/api/live/sync")
async def live_sync():
    return {"status": "success", "message": "Portfolio sync (paper)"}


@app.post("/api/mcp/analyze")
async def mcp_analyze(body: dict):
    sym = (body.get("symbol") or "RELIANCE.NS").upper()
    return {"symbol": sym, "analysis": {"trend": "bullish", "strength": random.uniform(0.6, 0.9), "recommendation": "BUY"}, "timestamp": datetime.now().isoformat()}


@app.post("/api/mcp/execute")
async def mcp_execute(body: dict):
    symbol = (body.get("symbol") or "").upper()
    side = (body.get("side") or "BUY").upper()
    qty = int(body.get("quantity", 0))
    return {"status": "success", "message": f"Paper order: {side} {qty} {symbol}", "order_id": f"paper-{datetime.now().strftime('%Y%m%d%H%M%S')}"}


@app.post("/api/mcp/chat")
async def mcp_chat(body: dict):
    msg = (body.get("message") or "").strip()
    return {"response": f"MCP chat: {msg[:100]}", "timestamp": datetime.now().isoformat()}


@app.get("/api/mcp/status")
async def mcp_status():
    return {"mcp_available": True, "server_initialized": True}


@app.post("/api/order")
async def place_order(order: OrderRequest):
    mode = bot_state.get("config", {}).get("mode", "paper")
    if mode == "live" and place_order_market and get_dhan_token and get_dhan_token():
        result = place_order_market(
            order.symbol, order.side, order.quantity,
            product_type="CNC",
            trigger_price=float(order.price) if order.price else None,
        )
        if result and result.get("orderId"):
            return {"status": "success", "order_id": result.get("orderId", ""), "message": "Order placed (Dhan)"}
        if result is None:
            raise HTTPException(status_code=400, detail="Could not resolve symbol or place order (check symbol in holdings or Dhan API)")
        raise HTTPException(status_code=502, detail=result.get("message", "Dhan order failed"))
    ts = datetime.now().isoformat()
    total = order.quantity * (order.price or 0)
    if "tradeLog" not in bot_state["portfolio"]:
        bot_state["portfolio"]["tradeLog"] = []
    bot_state["portfolio"]["tradeLog"].insert(0, {
        "timestamp": ts, "symbol": order.symbol.upper(), "action": order.side.upper(),
        "quantity": order.quantity, "price": order.price or 0, "total": total,
    })
    return {"status": "success", "order_id": f"paper-{ts}", "message": "Order placed (paper)"}


@app.get("/api/production/signal-performance")
async def signal_performance():
    return {"signals": [], "message": "Stub"}


@app.get("/api/production/risk-metrics")
async def risk_metrics():
    return {"metrics": {}, "message": "Stub"}


@app.post("/api/production/make-decision")
async def make_decision(body: dict):
    return {"decision": "HOLD", "symbol": body.get("symbol", ""), "message": "Stub"}


@app.get("/api/production/learning-insights")
async def learning_insights():
    return {"insights": [], "message": "Stub"}


@app.get("/api/production/decision-history")
async def decision_history(days: int = 7):
    return {"history": [], "days": days}


if __name__ == "__main__":
    import sys
    import os
    # One-command mode: run full HFT2 stack (fyers_data_service, start_mcp_server, web_backend, then this API)
    if os.environ.get("HFT2_SIMPLE_ONLY") != "1":
        run_hft2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_hft2.py")
        if os.path.isfile(run_hft2):
            os.execv(sys.executable, [sys.executable, run_hft2] + sys.argv[1:])
    port = 5001
    if "--port" in sys.argv:
        try:
            i = sys.argv.index("--port")
            port = int(sys.argv[i + 1])
        except (IndexError, ValueError):
            pass
    logger.info("Starting HFT2 API on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
