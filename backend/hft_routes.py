#!/usr/bin/env python3
"""
Minimal FastAPI Backend for HFT Trading Bot Frontend
Provides mock data endpoints for testing frontend integration
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class BotConfig(BaseModel):
    mode: str  # "paper" or "live"
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    max_allocation: float
    stop_loss: Optional[float] = None

class ChatMessage(BaseModel):
    message: str

# Initialize FastAPI app
app = FastAPI(title="HFT Trading Bot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data store
bot_state = {
    "isRunning": False,
    "config": {
        "mode": "paper",
        "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "riskLevel": "MEDIUM",
        "maxAllocation": 0.25,
        "stopLoss": 0.05
    },
    "portfolio": {
        "totalValue": 1050000,
        "cash": 850000,
        "startingBalance": 1000000,
        "holdings": {
            "RELIANCE.NS": {
                "symbol": "RELIANCE.NS",
                "quantity": 50,
                "avgPrice": 2500,
                "currentPrice": 2600,
                "lastAction": "BUY"
            },
            "TCS.NS": {
                "symbol": "TCS.NS",
                "quantity": 100,
                "avgPrice": 3200,
                "currentPrice": 3300,
                "lastAction": "BUY"
            }
        },
        "tradeLog": [
            {
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "symbol": "RELIANCE.NS",
                "action": "BUY",
                "quantity": 50,
                "price": 2500,
                "total": 125000
            },
            {
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
                "symbol": "TCS.NS",
                "action": "BUY",
                "quantity": 100,
                "price": 3200,
                "total": 320000
            }
        ]
    },
    "chatMessages": []
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "HFT Trading Bot API - Running", "status": "operational"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/bot-data")
async def get_bot_data():
    """Get complete bot data"""
    return bot_state

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio data"""
    return bot_state["portfolio"]

@app.get("/api/trades")
async def get_trades():
    """Get trade history"""
    return {"trades": bot_state["portfolio"]["tradeLog"]}

@app.post("/api/bot/start")
async def start_bot():
    """Start the trading bot"""
    bot_state["isRunning"] = True
    logger.info("Bot started")
    return {"status": "success", "message": "Bot started successfully", "isRunning": True}

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot"""
    bot_state["isRunning"] = False
    logger.info("Bot stopped")
    return {"status": "success", "message": "Bot stopped successfully", "isRunning": False}

@app.post("/api/settings")
async def update_settings(config: BotConfig):
    """Update bot settings"""
    try:
        bot_state["config"]["mode"] = config.mode
        bot_state["config"]["riskLevel"] = config.risk_level
        bot_state["config"]["maxAllocation"] = config.max_allocation
        if config.stop_loss:
            bot_state["config"]["stopLoss"] = config.stop_loss
        
        logger.info(f"Settings updated: {config.dict()}")
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/add/{ticker}")
async def add_to_watchlist(ticker: str):
    """Add ticker to watchlist"""
    if ticker not in bot_state["config"]["tickers"]:
        bot_state["config"]["tickers"].append(ticker)
        logger.info(f"Added {ticker} to watchlist")
    return {"status": "success", "message": f"Added {ticker} to watchlist", "tickers": bot_state["config"]["tickers"]}

@app.delete("/api/watchlist/remove/{ticker}")
async def remove_from_watchlist(ticker: str):
    """Remove ticker from watchlist"""
    if ticker in bot_state["config"]["tickers"]:
        bot_state["config"]["tickers"].remove(ticker)
        logger.info(f"Removed {ticker} from watchlist")
    return {"status": "success", "message": f"Removed {ticker} from watchlist", "tickers": bot_state["config"]["tickers"]}

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Send chat message to AI assistant"""
    user_msg = {
        "role": "user",
        "content": message.message,
        "timestamp": datetime.now().isoformat()
    }
    bot_state["chatMessages"].append(user_msg)
    
    # Generate mock response based on message content
    response_content = "I'm the HFT trading assistant. How can I help you today?"
    
    if "pnl" in message.message.lower() or "profit" in message.message.lower():
        total_value = bot_state["portfolio"]["totalValue"]
        starting = bot_state["portfolio"]["startingBalance"]
        pnl = total_value - starting
        pnl_pct = (pnl / starting) * 100
        response_content = f"Current P&L: ₹{pnl:,.2f} ({pnl_pct:.2f}%)\nTotal Portfolio Value: ₹{total_value:,.2f}"
    
    elif "position" in message.message.lower():
        holdings_count = len(bot_state["portfolio"]["holdings"])
        response_content = f"You have {holdings_count} active positions:\n"
        for symbol, holding in bot_state["portfolio"]["holdings"].items():
            pnl = (holding["currentPrice"] - holding["avgPrice"]) * holding["quantity"]
            response_content += f"- {symbol}: {holding['quantity']} shares, P&L: ₹{pnl:,.2f}\n"
    
    elif "start" in message.message.lower():
        response_content = "To start the bot, click the 'Start Bot' button or use the /start_bot command."
    
    assistant_msg = {
        "role": "assistant",
        "content": response_content,
        "timestamp": datetime.now().isoformat()
    }
    bot_state["chatMessages"].append(assistant_msg)
    
    return {"response": response_content, "messages": bot_state["chatMessages"]}

@app.get("/api/live-status")
async def get_live_status():
    """Get live trading status"""
    return {
        "connected": bot_state["isRunning"],
        "mode": bot_state["config"]["mode"],
        "lastUpdate": datetime.now().isoformat()
    }

@app.post("/api/mcp/analyze")
async def mcp_analyze(symbol: str):
    """Market Control Protocol analysis"""
    # Mock MCP analysis
    return {
        "symbol": symbol,
        "analysis": {
            "trend": "bullish",
            "strength": random.uniform(0.6, 0.9),
            "recommendation": "BUY"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import sys
    
    # Parse port from command line arguments
    port = 5001
    if "--port" in sys.argv:
        try:
            port_idx = sys.argv.index("--port")
            port = int(sys.argv[port_idx + 1])
        except (IndexError, ValueError):
            logger.warning("Invalid port argument, using default 5001")
    
    logger.info(f"Starting HFT Trading Bot API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
