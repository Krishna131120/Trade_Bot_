"""
HFT Trading Bot Routes
Integrated into main backend API server
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
import random

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Create router for HFT endpoints
hft_router = APIRouter()

# Pydantic models
class BotConfig(BaseModel):
    mode: str  # "paper" or "live"
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    max_allocation: float
    stop_loss: Optional[float] = None

class ChatMessage(BaseModel):
    message: str

# Mock data store - In production, this would be a database
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

# Health check
@hft_router.get("/health")
async def health_check():
    """Health check endpoint for HFT bot"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Status (alias for frontend getStatus() - same as health)
@hft_router.get("/status")
async def status():
    """Status endpoint for HFT bot (frontend compatibility)"""
    return {"status": "healthy", "isRunning": bot_state["isRunning"], "timestamp": datetime.now().isoformat()}

# Bot data
@hft_router.get("/bot-data")
async def get_bot_data():
    """Get complete bot data"""
    return bot_state

# Portfolio
@hft_router.get("/portfolio")
async def get_portfolio():
    """Get portfolio data"""
    return bot_state["portfolio"]

# Trades
@hft_router.get("/trades")
async def get_trades():
    """Get trade history"""
    return {"trades": bot_state["portfolio"]["tradeLog"]}

# Bot control - Start
@hft_router.post("/bot/start")
async def start_bot():
    """Start the trading bot"""
    bot_state["isRunning"] = True
    logger.info("HFT Bot started")
    return {"status": "success", "message": "Bot started successfully", "isRunning": True}

# Bot control - Stop
@hft_router.post("/bot/stop")
async def stop_bot():
    """Stop the trading bot"""
    bot_state["isRunning"] = False
    logger.info("HFT Bot stopped")
    return {"status": "success", "message": "Bot stopped successfully", "isRunning": False}

# Settings
@hft_router.post("/settings")
async def update_settings(config: BotConfig):
    """Update bot settings"""
    try:
        bot_state["config"]["mode"] = config.mode
        bot_state["config"]["riskLevel"] = config.risk_level
        bot_state["config"]["maxAllocation"] = config.max_allocation
        if config.stop_loss:
            bot_state["config"]["stopLoss"] = config.stop_loss
        
        logger.info(f"HFT Bot settings updated: {config.dict()}")
        return {"status": "success", "message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating HFT bot settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Watchlist - Add
@hft_router.post("/watchlist/add/{ticker}")
async def add_to_watchlist(ticker: str):
    """Add ticker to watchlist"""
    if ticker not in bot_state["config"]["tickers"]:
        bot_state["config"]["tickers"].append(ticker)
        logger.info(f"Added {ticker} to HFT watchlist")
    return {"status": "success", "message": f"Added {ticker} to watchlist", "tickers": bot_state["config"]["tickers"]}

# Watchlist - Remove
@hft_router.delete("/watchlist/remove/{ticker}")
async def remove_from_watchlist(ticker: str):
    """Remove ticker from watchlist"""
    if ticker in bot_state["config"]["tickers"]:
        bot_state["config"]["tickers"].remove(ticker)
        logger.info(f"Removed {ticker} from HFT watchlist")
    return {"status": "success", "message": f"Removed {ticker} from watchlist", "tickers": bot_state["config"]["tickers"]}

# Chat
@hft_router.post("/chat")
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

# Live status
@hft_router.get("/live-status")
async def get_live_status():
    """Get live trading status"""
    return {
        "connected": bot_state["isRunning"],
        "mode": bot_state["config"]["mode"],
        "lastUpdate": datetime.now().isoformat()
    }

# MCP analysis
@hft_router.post("/mcp/analyze")
async def mcp_analyze(symbol: str):
    """Market Control Protocol analysis"""
    return {
        "symbol": symbol,
        "analysis": {
            "trend": "bullish",
            "strength": random.uniform(0.6, 0.9),
            "recommendation": "BUY"
        },
        "timestamp": datetime.now().isoformat()
    }
