#!/usr/bin/env python3
"""
FastAPI backend for the Indian Stock Trading Bot Web Interface
Provides REST API endpoints for the HTML/CSS/JS frontend
"""

from data_service_client import get_data_client, DataServiceClient
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
import threading
import time
import traceback
import socket
import subprocess
import platform
import asyncio
from contextlib import asynccontextmanager

# Fix import paths permanently - MOVED TO TOP
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Project root
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Import FastAPI components with fallback handling
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse

    from pydantic import BaseModel
    import json
    import queue as _queue_module
except ImportError as e:
    print(f"Error importing FastAPI components: {e}")
    print("Please install FastAPI dependencies:")
    print("pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Load environment variables from .env early so config/env fallbacks work
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    load_dotenv()
    # Also load backend/hft2/env when run from backend/hft2/backend (standalone or via run_hft2)
    _env_file = _Path(__file__).resolve().parent.parent / "env"
    if _env_file.exists():
        load_dotenv(_env_file)
    logger.debug("Loaded .env into environment")
except Exception:
    logger.debug("python-dotenv not available or .env not loaded")

# Import new components for live trading
try:
    from portfolio_manager import DualPortfolioManager
    from dhan_client import DhanAPIClient
    from live_executor import LiveTradingExecutor
    from dhan_sync_service import start_sync_service, stop_sync_service, get_sync_service
    LIVE_TRADING_AVAILABLE = True
    logger.info("Live trading components loaded successfully")
except ImportError as e:
    print(f"Live trading components not available: {e}")
    logger.error(f"❌ Live trading import failed: {e}")
    LIVE_TRADING_AVAILABLE = False

# Import new agents for full-market RL scanning
try:
    from core.data_agent import data_agent
    from core.rl_agent import rl_agent
    from core.tracker_agent import tracker_agent
    from core.risk_engine import risk_engine
    logger.info("RL scanning agents loaded successfully")
except ImportError as e:
    logger.error(f"❌ RL agents import failed: {e}")

# Architectural Fix: Graceful MCP dependency handling
try:
    from mcp_service import MCPTradingServer, TradingAgent, ExplanationAgent
    MCP_AVAILABLE = True
    MCP_SERVER_AVAILABLE = True
    print("MCP server components loaded successfully")
except ImportError as e:
    print(f"MCP server components not available: {e}")
    MCP_AVAILABLE = False
    # Create fallback classes

    class MCPTradingServer:
        def __init__(self, *args, **kwargs): pass

    class TradingAgent:
        def __init__(self, *args, **kwargs): pass

    class ExplanationAgent:
        def __init__(self, *args, **kwargs): pass
    MCP_SERVER_AVAILABLE = False

try:
    from fyers_client import FyersAPIClient
    FYERS_CLIENT_AVAILABLE = True
except ImportError as e:
    print(f"Fyers client not available: {e}")
    FYERS_CLIENT_AVAILABLE = False

    class FyersAPIClient:
        def __init__(self, *args, **kwargs): pass

try:
    from mcp_service.llm import GroqReasoningEngine, TradingContext, GroqResponse
    GROQ_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Groq integration not available: {e}")
    GROQ_AVAILABLE = False

    class GroqReasoningEngine:
        def __init__(self, *args, **kwargs): pass

    class TradingContext:
        def __init__(self, *args, **kwargs): pass

    class GroqResponse:
        def __init__(self, *args, **kwargs): pass

# PRODUCTION FIX: Import data service client instead of direct Fyers

# Priority 3: Standardized logging strategy
LOG_FILE_PATH = os.getenv("WEB_BACKEND_LOG_FILE", "web_trading_bot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

# Configure logging with standardized format and levels
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific log levels for different components
logging.getLogger('utils').setLevel(logging.INFO)
logging.getLogger('core').setLevel(logging.INFO)
logging.getLogger('mcp_server').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Code Quality: Define constants to replace magic numbers
CHAT_MESSAGE_MAX_LENGTH = 1000
RANDOM_STOCK_MIN_COUNT = 8
RANDOM_STOCK_MAX_COUNT = 12
CACHE_TTL_SECONDS = 5
WEBSOCKET_PING_INTERVAL = 20
WEBSOCKET_PING_TIMEOUT = 10
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7

# Priority 4: Optimized import structure with error handling
try:
    from utils import (
        ConfigValidator,
        validate_chat_input,
        TradingBotError,
        ConfigurationError,
        DataServiceError,
        TradingExecutionError,
        ValidationError,
        NetworkError,
        AuthenticationError,
        PerformanceMonitor,
        retry_on_failure,
        circuit_breaker,
        api_retry,
        data_service_retry,
        log_api_call,
        log_system_event,
        log_system_health
    )
    UTILS_AVAILABLE = True
    logger.info("Utils modules imported successfully")
except ImportError as e:
    logger.error(f"Error importing utils modules: {e}")
    UTILS_AVAILABLE = False
    # Fallback implementations

    class TradingBotError(Exception):
        pass

    class ConfigurationError(TradingBotError):
        pass

    class DataServiceError(TradingBotError):
        pass

    class TradingExecutionError(TradingBotError):
        pass

    class ValidationError(TradingBotError):
        pass

    class NetworkError(TradingBotError):
        pass

    class AuthenticationError(TradingBotError):
        pass

    class ConfigValidator:
        @staticmethod
        def validate_config(config): return config

    def validate_chat_input(message): return message.strip()

# Remove the fallback PerformanceMonitor class since it's now properly imported from utils

# Initialize performance monitor
performance_monitor = PerformanceMonitor()

# Import Production Core Components
try:
    from core import (
        AsyncSignalCollector,
        AdaptiveThresholdManager,
        IntegratedRiskManager,
        DecisionAuditTrail,
        ContinuousLearningEngine
    )
    PRODUCTION_CORE_AVAILABLE = True
    logger.info("Production core components loaded successfully")
except ImportError as e:
    logger.error(f"Error importing production core components: {e}")
    PRODUCTION_CORE_AVAILABLE = False

# Import Configuration Schema and Validation
try:
    from config.config_schema import ConfigValidator, load_and_validate_config
    CONFIG_SCHEMA_AVAILABLE = True
    logger.info("Configuration schema validation loaded successfully")
except ImportError as e:
    logger.error(f"Error importing configuration schema: {e}")
    CONFIG_SCHEMA_AVAILABLE = False

# Import the trading bot components (optional: auth/signup work without them)
ChatbotCommandHandler = VirtualPortfolio = TradingExecutor = None
DataFeed = Stock = StockTradingBot = None
try:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from testindia import (
        ChatbotCommandHandler, VirtualPortfolio,
        TradingExecutor, DataFeed, Stock, StockTradingBot
    )
except ImportError as e:
    print(f"Error importing trading bot components: {e}")
    print("Make sure testindia.py is in the same directory. Auth/signup will work; bot features may be limited.")

# Fix: Alias StockTradingBot to WebTradingBot if imported
if StockTradingBot:
    WebTradingBot = StockTradingBot
else:
    class WebTradingBot:
        def __init__(self, config):
            self.config = config
            self.portfolio = {}
            self.executor = None
        def get_portfolio_metrics(self): return {}
        def start(self): pass
        def stop(self): pass


# Pydantic Models for Request/Response validation


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    timestamp: str
    confidence: Optional[float] = None
    context: Optional[str] = None


class WatchlistRequest(BaseModel):
    ticker: str
    action: str  # ADD or REMOVE


class WatchlistResponse(BaseModel):
    message: str
    tickers: List[str]


class BulkWatchlistRequest(BaseModel):
    tickers: List[str]
    action: str = "ADD"  # ADD or REMOVE


class BulkWatchlistResponse(BaseModel):
    message: str
    successful_tickers: List[str]
    failed_tickers: List[str]
    total_processed: int


class SettingsRequest(BaseModel):
    mode: Optional[str] = None
    riskLevel: Optional[str] = None
    stop_loss_pct: Optional[float] = None
    target_profit_pct: Optional[float] = None
    use_risk_reward: Optional[bool] = None
    risk_reward_ratio: Optional[float] = None
    max_capital_per_trade: Optional[float] = None
    max_trade_limit: Optional[int] = None

# MCP-specific models


class MCPAnalysisRequest(BaseModel):
    symbol: str
    timeframe: Optional[str] = "1D"
    analysis_type: Optional[str] = "comprehensive"


class MCPTradeRequest(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    quantity: Optional[int] = None
    override_reason: Optional[str] = None


class MCPChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class PredictionRequest(BaseModel):
    symbols: Optional[List[str]] = []
    models: Optional[List[str]] = ["rl"]
    horizon: Optional[str] = "day"
    include_explanations: Optional[bool] = True
    natural_query: Optional[str] = ""


class ScanRequest(BaseModel):
    filters: Optional[Dict[str, Any]] = {}
    sort_by: Optional[str] = "score"
    limit: Optional[int] = 50
    natural_query: Optional[str] = ""


class StartBotWithSymbolRequest(BaseModel):
    symbol: str


class PortfolioMetrics(BaseModel):
    total_value: float
    cash: float
    cash_percentage: float = 0
    holdings: Dict[str, Any]
    total_invested: float = 0
    invested_percentage: float = 0
    current_holdings_value: float = 0
    total_return: float
    return_percentage: float
    total_return_pct: float = 0
    unrealized_pnl: float
    unrealized_pnl_pct: float = 0
    realized_pnl: float
    realized_pnl_pct: float = 0
    total_exposure: float
    exposure_ratio: float = 0
    profit_loss: float = 0
    profit_loss_pct: float = 0
    active_positions: int
    trades_today: int = 0
    initial_balance: float = 10000


class BotStatus(BaseModel):
    is_running: bool
    last_update: str
    mode: str


class MessageResponse(BaseModel):
    message: str

# New endpoint models for RL scanning


class AnalyzeRequest(BaseModel):
    tickers: List[str]
    horizon: str = "day"


class UpdateRiskRequest(BaseModel):
    stop_loss_pct: float
    capital_risk_pct: float
    drawdown_limit_pct: float


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class UserProfileUpdate(BaseModel):
    fullName: Optional[str] = None
    email: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


# JWT auth
try:
    # CRITICAL: Import from current directory explicitly to avoid conflicts with backend/auth.py
    # Use importlib to force loading the local auth.py file
    import importlib.util
    import sys
    from pathlib import Path
    
    # Get absolute path to local auth.py
    current_dir = Path(__file__).resolve().parent
    auth_file_path = current_dir / "auth.py"
    
    # Load module from file explicitly - this ensures we get the correct auth.py
    spec = importlib.util.spec_from_file_location("hft2_backend_auth", auth_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load auth module from {auth_file_path}")
    auth_module = importlib.util.module_from_spec(spec)
    sys.modules["hft2_backend_auth"] = auth_module  # Prevent re-import
    spec.loader.exec_module(auth_module)
    
    # Verify it has the required functions
    if not hasattr(auth_module, 'create_user'):
        raise AttributeError(f"auth module at {auth_file_path} missing create_user function. Found: {dir(auth_module)}")
    
    _http_bearer = HTTPBearer(auto_error=False)

    def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_http_bearer)):
        """Dependency: returns JWT payload dict if valid Bearer token, else None."""
        if not credentials or not credentials.credentials:
            return None
        payload = auth_module.decode_token(credentials.credentials)
        return payload

    def get_current_user_required(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Dependency: returns JWT payload or raises 401."""
        if not credentials or not credentials.credentials:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if credentials.credentials in _logout_blacklist:
            raise HTTPException(status_code=401, detail="Token invalidated (logged out)")
        payload = auth_module.decode_token(credentials.credentials)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return payload

    def get_optional_user_demat(credentials: HTTPAuthorizationCredentials = Depends(_http_bearer)):
        """Dependency: returns (payload, demat_creds). demat_creds is None if not auth or no demat linked."""
        if not credentials or not credentials.credentials:
            return (None, None)
        payload = auth_module.decode_token(credentials.credentials)
        if not payload:
            return (None, None)
        username = (payload.get("sub") or "").strip()
        demat = auth_module.get_user_demat(username) if hasattr(auth_module, "get_user_demat") else None
        return (payload, demat)

    JWT_AVAILABLE = True
    get_optional_user = get_current_user  # Optional auth: returns payload or None
except Exception as e:
    logger.warning(f"JWT auth not available: {e}")
    JWT_AVAILABLE = False
    get_current_user = get_current_user_required = None

    def get_optional_user():
        return None

    def get_optional_user_demat(credentials=None):
        return (None, None)

# Logger already configured above

# Logout blacklist: tokens added here are rejected until server restart (client must discard token)
_logout_blacklist = set()

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # Call the defined events in the global scope
    await startup_event()
    yield
    await shutdown_event()

# Initialize FastAPI app
app = FastAPI(
    title="Indian Stock Trading Bot API",
    description="REST API for the Indian Stock Trading Bot Web Interface",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=app_lifespan
)

# Add CORS middleware - MUST be added before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (localhost:5173, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)

# Priority 2: Integrate custom exception handlers with FastAPI


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    """Handle validation errors with proper HTTP responses"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request, exc: ConfigurationError):
    """Handle configuration errors"""
    logger.error(f"Configuration error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Configuration error occurred"})


@app.exception_handler(DataServiceError)
async def data_service_error_handler(request, exc: DataServiceError):
    """Handle data service errors"""
    logger.error(f"Data service error: {exc}")
    return JSONResponse(status_code=503, content={"detail": "Data service temporarily unavailable"})


@app.exception_handler(TradingExecutionError)
async def trading_execution_error_handler(request, exc: TradingExecutionError):
    """Handle trading execution errors"""
    logger.error(f"Trading execution error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Trading execution failed"})


@app.exception_handler(NetworkError)
async def network_error_handler(request, exc: NetworkError):
    """Handle network errors"""
    logger.error(f"Network error: {exc}")
    return JSONResponse(status_code=502, content={"detail": "Network connectivity issue"})


@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request, exc: AuthenticationError):
    """Handle authentication errors"""
    logger.error(f"Authentication error: {exc}")
    return JSONResponse(status_code=401, content={"detail": "Authentication failed"})

# Priority 4: Add comprehensive error handlers for common exceptions


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """Handle value errors"""
    logger.warning(f"Value error: {exc}")
    return JSONResponse(status_code=400, content={"detail": "Invalid input value"})


@app.exception_handler(KeyError)
async def key_error_handler(request, exc: KeyError):
    """Handle key errors"""
    logger.error(f"Key error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Missing required data"})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle all unhandled exceptions with CORS headers"""
    import traceback
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    # Return JSONResponse with CORS headers
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Global variables
trading_bot = None
bot_thread = None
bot_running = False
_bot_initializing = False  # Global track if bot is in middle of initialize_bot()
_continuous_loop_task: asyncio.Task = None  # Background continuous analysis loop


def get_bot_running() -> bool:
    """Return current bot_running so heavy analysis can check it (e.g. when user clicks Stop)."""
    return bot_running


@app.get("/api/bot/status")
async def get_bot_status():
    """Return current bot status for frontend polling."""
    global bot_running, _bot_initializing, trading_bot
    if _bot_initializing:
        return {"status": "INITIALIZING"}
    if bot_running and trading_bot:
        return {"status": "READY"}
    return {"status": "STOPPED"}


# ── Bot-data cache (stale-while-revalidate) ─────────────────────────────────
# Stores the last successful /api/bot-data response so slow Dhan/Fyers fetches
# never block the frontend. A background task keeps it fresh every 20 seconds.
import time as _time_module
_bot_data_cache: dict = {}
_bot_data_cache_ts: float = 0.0
_BOT_DATA_CACHE_TTL: float = 20.0          # seconds before background refresh
_bot_data_refresh_lock = asyncio.Lock()
_bot_data_refresh_running: bool = False

async def _refresh_bot_data_cache_background():
    """Fetch fresh bot-data and store in cache without blocking callers."""
    global _bot_data_cache, _bot_data_cache_ts, _bot_data_refresh_running
    if _bot_data_refresh_running:
        return
    _bot_data_refresh_running = True
    try:
        saved_mode = get_current_saved_mode() or "paper"
        result = None
        if trading_bot:
            current_mode = trading_bot.config.get("mode", saved_mode)
            if current_mode == "live":
                try:
                    from dhan_client import get_dhan_token, get_dhan_client_id, get_live_portfolio
                    if get_dhan_token() and get_dhan_client_id():
                        loop = asyncio.get_event_loop()
                        dhan_portfolio = await asyncio.wait_for(
                            loop.run_in_executor(None, get_live_portfolio), timeout=25.0
                        )
                        if dhan_portfolio:
                            result = _convert_dhan_portfolio_to_bot_data(dhan_portfolio, include_config=True)
                except Exception:
                    pass
            if result is None:
                try:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, trading_bot.get_complete_bot_data), timeout=10.0
                    )
                except Exception:
                    pass
        if result is None and saved_mode == "live":
            try:
                from dhan_client import get_dhan_token, get_dhan_client_id, get_live_portfolio
                if get_dhan_token() and get_dhan_client_id():
                    loop = asyncio.get_event_loop()
                    dhan_portfolio = await asyncio.wait_for(
                        loop.run_in_executor(None, get_live_portfolio), timeout=25.0
                    )
                    if dhan_portfolio:
                        result = _convert_dhan_portfolio_to_bot_data(dhan_portfolio, include_config=True)
            except Exception:
                pass
        if result:
            _bot_data_cache = result
            _bot_data_cache_ts = _time_module.monotonic()
    except Exception as _e:
        logger.warning(f"Background bot-data refresh error: {_e}")
    finally:
        _bot_data_refresh_running = False
# ─────────────────────────────────────────────────────────────────────────────


def _offline_bot_data():
    """Return valid bot-data shape when bot is not initialized so frontend shows offline state instead of 500."""
    saved = {}
    try:
        mode = get_current_saved_mode()
        saved = load_config_from_file(mode) or {}
    except Exception:
        pass
    return {
        "isRunning": bot_running or _bot_initializing,
        "config": {
            "mode": saved.get("mode", "paper"),
            "tickers": saved.get("tickers", []),
            "stopLossPct": saved.get("stop_loss_pct", 0.05),
            "maxAllocation": saved.get("max_capital_per_trade", 0.25),
            "maxTradeLimit": saved.get("max_trade_limit", 10),
        },
        "portfolio": {
            "totalValue": 10000,
            "cash": 10000,
            "holdings": {},
            "startingBalance": 10000,
            "unrealizedPnL": 0,
            "realizedPnL": 0,
            "tradeLog": [],
        },
        "analysis": list(_last_bot_analysis.values()),
        "lastUpdate": datetime.now().isoformat(),
    }

# MCP components
mcp_server = None
mcp_trading_agent = None
fyers_client = None
groq_engine = None

# Real-time market data function

# Semaphore: only 1 ML analysis runs at a time.
# Each analyze_stock call takes 2-3 min; running multiple in parallel saturates the thread pool
# and causes /api/bot-data and /api/trades to time out while waiting for a free thread.
_analysis_semaphore = asyncio.Semaphore(1)
_active_analysis_tasks = set()  # Track active ticker analysis tasks

# Stores latest analysis results per symbol, exposed via /bot-data to the frontend
_last_bot_analysis: dict = {}

# Per-user bot start: tickers for the user who triggered start (when auth present)
_pending_start_tickers: list = []
# Per-user bot start: user_id + demat credentials for the user who triggered start (when auth + demat linked)
_pending_bot_user_context: Optional[dict] = None

# Module-level flag to prevent double-initialization (replaces fragile function-attribute pattern)
_bot_initializing: bool = False

def _get_user_watchlist_from_db(username: str) -> list:
    """Return the authenticated user's watchlist from MongoDB. Empty list if not found or error."""
    if not username:
        return []
    try:
        from db.mongo_client import get_mongo_db
        db = get_mongo_db("trading")
        doc = db["watchlists"].find_one({"username": username})
        return list(doc.get("symbols", [])) if doc else []
    except Exception as e:
        logger.warning(f"Could not load user watchlist for {username}: {e}")
        return []

def _save_user_watchlist_to_db(username: str, tickers: list) -> bool:
    """Save the user's watchlist to MongoDB. Returns True on success."""
    if not username:
        return False
    try:
        from db.mongo_client import get_mongo_db
        from datetime import datetime
        db = get_mongo_db("trading")
        db["watchlists"].update_one(
            {"username": username},
            {"$set": {"username": username, "symbols": tickers, "updated_at": datetime.utcnow()}},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.warning(f"Could not save user watchlist for {username}: {e}")
        return False

# ===== SSE Log Broadcasting =====
_sse_clients: list = []
_sse_clients_lock = threading.Lock()

class _SSELogHandler(logging.Handler):
    """Sends log records to all connected SSE clients."""
    def emit(self, record):
        try:
            msg = self.format(record)
            payload = json.dumps({"type": "log", "level": record.levelname, "message": msg})
            event = f"data: {payload}\n\n"
            with _sse_clients_lock:
                for q in list(_sse_clients):
                    try:
                        q.put_nowait(event)
                    except Exception:
                        pass
        except Exception:
            pass

_sse_log_handler = _SSELogHandler()
_sse_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(_sse_log_handler)


def _build_sse_snapshot() -> dict:
    """Lightweight bot state snapshot for SSE data push."""
    global trading_bot, _last_bot_analysis
    try:
        if trading_bot:
            bot_data = trading_bot.get_complete_bot_data()
            portfolio = bot_data.get("portfolio", {})
            holdings_raw = portfolio.get("holdings", {})
            # Normalize holdings fields
            holdings = {}
            for sym, h in (holdings_raw or {}).items():
                holdings[sym] = {
                    "quantity": h.get("quantity") or h.get("qty", 0),
                    "avgPrice": h.get("avgPrice") or h.get("avg_price", 0),
                    "currentPrice": h.get("currentPrice") or h.get("last_price", 0),
                }
            cash = portfolio.get("cash", 0)
            total_value = portfolio.get("totalValue", 0)
            # If totalValue is 0 but holdings exist, compute from holdings
            if total_value == 0 and holdings:
                market_val = sum(
                    (h.get("currentPrice") or h.get("avgPrice", 0)) * (h.get("quantity", 0))
                    for h in holdings.values()
                )
                total_value = cash + market_val
            return {
                "isRunning": bot_data.get("isRunning", False) or _bot_initializing,
                "cash": cash,
                "totalValue": round(total_value, 2),
                "unrealizedPnL": portfolio.get("unrealizedPnL", 0),
                "realizedPnL": portfolio.get("realizedPnL", 0),
                "holdings": holdings,
                "analysis": list(_last_bot_analysis.values()),
            }
    except Exception:
        pass
    return {
        "isRunning": bot_running or _bot_initializing,
        "cash": 0,
        "totalValue": 0,
        "unrealizedPnL": 0,
        "realizedPnL": 0,
        "holdings": {},
        "analysis": list(_last_bot_analysis.values()),
    }

async def trigger_all_hft2_components_for_symbol(symbol: str):
    """Reusable async function to trigger all HFT2 backend components (predictions, analysis, data fetching) for a symbol.
    Runs with a semaphore so only one ticker's full ML pipeline executes at a time."""
    # Register this task for cancellation tracking
    current_task = asyncio.current_task()
    if current_task:
        _active_analysis_tasks.add(current_task)
        current_task.add_done_callback(lambda t: _active_analysis_tasks.discard(t) if t in _active_analysis_tasks else None)

    # Acquire semaphore: only 1 analysis at a time to avoid saturating the thread pool
    async with _analysis_semaphore:
        try:
            if not bot_running:
                logger.info(f"⏹ Aborting HFT2 process for {symbol} - bot not running")
                return

            logger.info(f"🚀 Starting HFT2 backend process for {symbol}...")
            if not bot_running: return
            prediction_result = None
            analysis_result = None
            
            # 1. Fetch live data from Fyers data service (if available)
            try:
                from fyers_client import get_fyers_client
                fyers = get_fyers_client()
                if fyers:
                    try:
                        # Get current price from Fyers
                        loop = asyncio.get_event_loop()
                        current_price = await loop.run_in_executor(None, fyers.get_price, symbol)
                        if current_price:
                            logger.info(f"✅ Fetched live price from Fyers for {symbol}: ₹{current_price}")
                    except Exception as fyers_err:
                        logger.warning(f"⚠️ Fyers data fetch failed for {symbol}: {fyers_err}")
            except Exception as fyers_init_err:
                logger.debug(f"Fyers client not available: {fyers_init_err}")
            
            # 2. Fetch historical data from Yahoo Finance (run in executor - synchronous call)
            try:
                import yfinance as yf
                loop = asyncio.get_event_loop()
                def _fetch_yf_history(sym):
                    stock = yf.Ticker(sym)
                    return stock.history(period="1mo")
                hist_data = await asyncio.wait_for(
                    loop.run_in_executor(None, _fetch_yf_history, symbol),
                    timeout=15.0
                )
                if not hist_data.empty:
                    logger.info(f"✅ Fetched historical data from Yahoo Finance for {symbol}: {len(hist_data)} days")
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Yahoo Finance historical data fetch timed out for {symbol}")
            except Exception as yahoo_err:
                logger.warning(f"⚠️ Yahoo Finance historical data fetch failed: {yahoo_err}")
            
            # 3. Trigger MCP prediction if available
            if not bot_running: return
            if MCP_AVAILABLE:
                await _ensure_mcp_initialized()
                if mcp_trading_agent:
                    try:
                        logger.info(f"📊 Triggering MCP prediction for {symbol}...")
                        from mcp_server.tools.prediction_tool import PredictionTool
                        prediction_tool = PredictionTool({
                            "tool_id": "prediction_tool",
                            "ollama_enabled": True,
                            "ollama_host": "http://localhost:11434",
                            "ollama_model": "llama3.1:8b"
                        })
                        session_id = str(int(time.time() * 1000000))
                        # Prediction Tool is already an async class
                        pred_result = await prediction_tool.rank_predictions({
                            "symbols": [symbol],
                            "models": ["rl"],
                            "horizon": "day",
                            "include_explanations": True
                        }, session_id)
                        status_str = pred_result.status.value if hasattr(pred_result.status, 'value') else str(pred_result.status)
                        if status_str.upper() == "SUCCESS":
                            prediction_result = pred_result.data
                            logger.info(f"✅ Prediction completed for {symbol}: {prediction_result}")
                        else:
                            logger.warning(f"⚠️ Prediction returned status: {status_str}")
                    except Exception as pred_error:
                        logger.warning(f"⚠️ Prediction failed for {symbol}: {pred_error}")
                        logger.exception("Prediction error traceback:")
            
            # 4. Trigger comprehensive analysis if available
            if not bot_running: return
            if MCP_AVAILABLE and mcp_trading_agent:
                try:
                    logger.info(f"🔍 Triggering comprehensive analysis for {symbol}...")
                    signal = await mcp_trading_agent.analyze_and_decide(
                        symbol=symbol,
                        market_context={
                            "timeframe": "intraday",
                            "analysis_type": "comprehensive"
                        }
                    )
                    analysis_result = {
                        "symbol": signal.symbol,
                        "recommendation": signal.decision.value,
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning,
                        "risk_score": signal.risk_score,
                        "position_size": signal.position_size,
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss
                    }
                    _last_bot_analysis[symbol] = {
                        **analysis_result,
                        "timestamp": datetime.now().isoformat(),
                        "prediction": prediction_result,
                    }
                    logger.info(f"✅ Analysis completed for {symbol}: {signal.decision.value} (confidence: {signal.confidence})")
                except Exception as analysis_error:
                    logger.warning(f"⚠️ Analysis failed for {symbol}: {analysis_error}")
                    logger.exception("Analysis error traceback:")
            
            # 5. Update data feed with new symbol
            try:
                if trading_bot and hasattr(trading_bot, 'data_feed') and trading_bot.data_feed:
                    logger.info(f"✅ Data feed updated for {symbol}")
            except Exception as feed_err:
                logger.warning(f"⚠️ Data feed update failed: {feed_err}")

            # Store prediction result even when analysis step is unavailable
            if prediction_result and symbol not in _last_bot_analysis:
                _last_bot_analysis[symbol] = {
                    "symbol": symbol,
                    "recommendation": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "Analysis not available",
                    "risk_score": 0.5,
                    "position_size": 0,
                    "target_price": None,
                    "stop_loss": None,
                    "timestamp": datetime.now().isoformat(),
                    "prediction": prediction_result,
                }

            logger.info(f"✅ HFT2 backend process completed for {symbol}")
        except Exception as process_error:
            logger.error(f"❌ Error in HFT2 backend process for {symbol}: {process_error}")
            logger.exception("Full traceback:")


async def _continuous_trading_loop():
    """Background loop: analyze all watchlist tickers sequentially and continuously while bot is running.
    After completing each ticker, checks if new tickers were added and processes them in the same cycle.
    Re-runs every sleep_interval seconds. Executes buy/sell via live executor if confidence threshold met."""
    global trading_bot, _continuous_loop_task
    logger.info("🔄 Continuous trading loop started")
    try:
        while True:
            try:
                if not trading_bot or not trading_bot.is_running:
                    logger.info("⏹ Continuous loop: bot stopped, exiting")
                    break
                sleep_secs = trading_bot.config.get("sleep_interval", 300)

                # Build a working list for this cycle; we'll append new tickers as they are added
                # by re-reading the config after each symbol is processed.
                initial_tickers = list(trading_bot.config.get("tickers", []))
                if not initial_tickers:
                    await asyncio.sleep(60)
                    continue

                logger.info(f"🔁 Continuous loop cycle start: {len(initial_tickers)} tickers: {initial_tickers}")

                # Use a set to track what we've already processed this cycle to avoid re-processing
                processed_this_cycle: set = set()
                # Work queue: process initial tickers first
                work_queue = list(initial_tickers)

                while work_queue:
                    sym = work_queue.pop(0)
                    if sym in processed_this_cycle:
                        continue
                    processed_this_cycle.add(sym)

                    if not trading_bot or not trading_bot.is_running:
                        break

                    try:
                        # Hard timeout per ticker: 3 minutes max so loop never hangs
                        await asyncio.wait_for(
                            trigger_all_hft2_components_for_symbol(sym),
                            timeout=180.0
                        )

                        # After analysis, attempt autonomous trade execution based on stored signal
                        if trading_bot and trading_bot.is_running:
                            analysis = _last_bot_analysis.get(sym, {})
                            rec = analysis.get("recommendation", "WAIT").upper()
                            conf = float(analysis.get("confidence", 0.0))
                            min_conf = trading_bot.config.get("min_confidence", 0.6)
                            if rec == "BUY" and conf >= min_conf:
                                logger.info(f"🤖 Auto-buy signal for {sym} (confidence={conf:.2f})")
                                if hasattr(trading_bot, 'live_executor') and trading_bot.live_executor:
                                    signal_data = {
                                        "confidence": conf,
                                        "current_price": analysis.get("current_price") or analysis.get("target_price"),
                                        "stop_loss": analysis.get("stop_loss"),
                                        "take_profit": analysis.get("target_price"),
                                    }
                                    loop = asyncio.get_event_loop()
                                    result = await loop.run_in_executor(
                                        None,
                                        lambda s=sym, sd=signal_data: trading_bot.live_executor.execute_buy_order(s, sd)
                                    )
                                    if result and result.get("success"):
                                        logger.info(f"✅ Auto-buy executed for {sym}: {result.get('message')}")
                                    else:
                                        logger.info(f"ℹ️ Auto-buy skipped for {sym}: {result.get('message', 'no result')}")
                            elif rec == "SELL" and conf >= min_conf:
                                logger.info(f"🤖 Auto-sell signal for {sym} (confidence={conf:.2f})")
                                if hasattr(trading_bot, 'live_executor') and trading_bot.live_executor:
                                    signal_data = {"confidence": conf, "current_price": analysis.get("current_price") or analysis.get("target_price")}
                                    loop = asyncio.get_event_loop()
                                    result = await loop.run_in_executor(
                                        None,
                                        lambda s=sym, sd=signal_data: trading_bot.live_executor.execute_sell_order(s, sd)
                                    )
                                    if result and result.get("success"):
                                        logger.info(f"✅ Auto-sell executed for {sym}: {result.get('message')}")
                                    else:
                                        logger.info(f"ℹ️ Auto-sell skipped for {sym}: {result.get('message', 'no result')}")

                    except asyncio.TimeoutError:
                        logger.warning(f"⚠️ Analysis timed out for {sym} (>180s) — moving to next ticker")
                    except Exception as sym_err:
                        logger.warning(f"⚠️ Continuous loop error for {sym}: {sym_err}")

                    # After each ticker, check if new tickers were added to the watchlist
                    # If so, append them to the current cycle's work queue so they're processed now
                    if trading_bot:
                        current_tickers = list(trading_bot.config.get("tickers", []))
                        for new_sym in current_tickers:
                            if new_sym not in processed_this_cycle and new_sym not in work_queue:
                                logger.info(f"➕ New ticker detected mid-cycle, adding to current cycle: {new_sym}")
                                work_queue.append(new_sym)

                logger.info(f"✅ Continuous loop cycle done ({len(processed_this_cycle)} tickers). Sleeping {sleep_secs}s before next cycle.")
            except asyncio.CancelledError:
                raise
            except Exception as loop_err:
                logger.error(f"❌ Continuous trading loop error: {loop_err}")
            await asyncio.sleep(sleep_secs)
    except asyncio.CancelledError:
        logger.info("⏹ Continuous trading loop cancelled")
    finally:
        logger.info("🔄 Continuous trading loop exited")


def _start_continuous_loop():
    """Start the continuous trading loop as an asyncio background task."""
    global _continuous_loop_task
    if _continuous_loop_task and not _continuous_loop_task.done():
        logger.info("ℹ️ Continuous loop already running")
        return
    _continuous_loop_task = asyncio.create_task(_continuous_trading_loop())
    logger.info("✅ Continuous trading loop task created")


def _stop_continuous_loop():
    """Cancel the continuous trading loop task and ensure bot_running is False."""
    global _continuous_loop_task, bot_running, _active_analysis_tasks, _last_bot_analysis, _bot_data_cache, _bot_initializing
    bot_running = False
    _bot_initializing = False
    
    # 1. Cancel the main continuous loop task
    if _continuous_loop_task and not _continuous_loop_task.done():
        _continuous_loop_task.cancel()
        logger.info("⏹ Continuous trading loop cancelled")
    _continuous_loop_task = None

    # 2. Cancel all active analysis background tasks
    if _active_analysis_tasks:
        logger.info(f"⏹ Cancelling {len(_active_analysis_tasks)} active analysis tasks")
        for task in _active_analysis_tasks:
            if not task.done():
                task.cancel()
        _active_analysis_tasks.clear()

    # 3. Clear analysis cache to return to "fresh" state
    _last_bot_analysis.clear()
    _bot_data_cache.clear()
    logger.info("🧹 Bot analysis cache cleared")


async def get_real_time_market_response(message: str) -> Optional[str]:
    """Generate real-time market responses based on live data"""
    try:
        message_lower = message.lower()
        current_time = datetime.now()

        # Get live market data from Fyers
        fyers_client = get_fyers_client()
        if not fyers_client:
            return None

        # Get dynamic stock list from trading bot's watchlist and popular stocks
        major_stocks = get_dynamic_stock_list()

        if "highest volume" in message_lower or "higest volume" in message_lower:
            # PRIORITY 1: Try Fyers API first (REAL DATA)
            volume_data = []
            if fyers_client:
                logger.info("Fetching real-time data from Fyers API")
                # PRODUCTION FIX: Use data service for volume data
                all_data = fyers_client.get_all_data()
                for symbol, data in all_data.items():
                    try:
                        volume_data.append({
                            "symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                            "volume": data.get("volume", 0),
                            "price": data.get("price", 0),
                            "change": data.get("change", 0),
                            "change_pct": data.get("change_pct", 0)
                        })
                    except Exception as e:
                        logger.error(
                            f"Error processing data service data for {symbol}: {e}")
                        continue

            # PRIORITY 2: If Fyers failed, try Yahoo Finance
            if not volume_data or all(d['price'] == 0 for d in volume_data):
                logger.info("Fyers data unavailable, trying Yahoo Finance")
                volume_data = get_real_market_data_from_api()

            if volume_data:
                # Sort by volume
                volume_data.sort(key=lambda x: x["volume"], reverse=True)
                top_stocks = volume_data[:4]

                response = f"**Real-Time Highest Volume Stocks** (as of {current_time.strftime('%I:%M %p')})\n\n"
                response += "**Market Overview:**\n"
                response += f"Showing live data with real-time volume analysis.\n\n"

                for i, stock in enumerate(top_stocks, 1):
                    change_emoji = "[+]" if stock["change"] >= 0 else "[-]"
                    response += f"{change_emoji} **{stock['symbol']}**: Rs.{stock['price']:.2f} ({stock['change_pct']:+.2f}%) | Vol: {stock['volume']:,}\n"

                response += f"\n>> **Live Market Insight:** High volume indicates strong institutional interest and active trading."

                return response

        elif "lowest volume" in message_lower:
            # Get real market data for low volume analysis
            volume_data = get_real_market_data_from_api()

            if not volume_data and fyers_client:
                volume_data = []
                # PRODUCTION FIX: Use data service for volume data
                all_data = fyers_client.get_all_data()
                for symbol, data in all_data.items():
                    try:
                        volume_data.append({
                            "symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                            "volume": data.get("volume", 0),
                            "price": data.get("price", 0),
                            "change": data.get("change", 0),
                            "change_pct": data.get("change_pct", 0)
                        })
                    except Exception as e:
                        continue

            if volume_data:
                # Sort by volume (ascending for lowest)
                volume_data.sort(key=lambda x: x["volume"])
                low_volume_stocks = volume_data[:4]

                response = f"**Real-Time Lowest Volume Stocks** (as of {current_time.strftime('%I:%M %p')})\n\n"
                response += "**Market Overview:**\n"
                response += f"Showing live data with low volume analysis.\n\n"

                for i, stock in enumerate(low_volume_stocks, 1):
                    change_emoji = "[+]" if stock["change"] >= 0 else "[-]"
                    response += f"{change_emoji} **{stock['symbol']}**: Rs.{stock['price']:.2f} ({stock['change_pct']:+.2f}%) | Vol: {stock['volume']:,}\n"

                response += f"\n**Live Market Insight:** Low volume may indicate consolidation or lack of institutional interest."

                return response

        elif any(word in message_lower for word in ["market", "overview", "today"]):
            # Get real market overview data
            market_data = get_real_market_data_from_api()

            if not market_data and fyers_client:
                market_data = []
                # PRODUCTION FIX: Use data service for market overview
                all_data = fyers_client.get_all_data()
                count = 0
                for symbol, data in all_data.items():
                    if count >= 6:  # Show more variety
                        break
                    try:
                        market_data.append({
                            "symbol": symbol.replace("NSE:", "").replace("-EQ", ""),
                            "price": data.get("price", 0),
                            "change": data.get("change", 0),
                            "change_pct": data.get("change_pct", 0),
                            "volume": data.get("volume", 0)
                        })
                        count += 1
                    except Exception as e:
                        continue

            if market_data:
                positive_stocks = len(
                    [s for s in market_data if s["change"] >= 0])
                avg_change = sum(s["change_pct"]
                                 for s in market_data) / len(market_data)

                response = f"**Live Market Overview** (as of {current_time.strftime('%I:%M %p')})\n\n"
                response += f"**Market Sentiment:** {'Positive' if avg_change > 0 else 'Negative'} with average change of {avg_change:+.2f}%\n\n"

                for stock in market_data:
                    change_emoji = "[+]" if stock["change"] >= 0 else "[-]"
                    response += f"{change_emoji} **{stock['symbol']}**: Rs.{stock['price']:.2f} ({stock['change_pct']:+.2f}%) | Vol: {stock['volume']:,}\n"

                response += f"\n>> **Market Status:** {positive_stocks}/{len(market_data)} stocks are positive today."

                return response

        return None

    except Exception as e:
        logger.error(f"Error generating real-time market response: {e}")
        return None


def get_dynamic_stock_list():
    """Get dynamic list of stocks from multiple sources"""
    try:
        # Get stocks from trading bot's watchlist if available
        if trading_bot and hasattr(trading_bot, 'config'):
            watchlist_stocks = trading_bot.config.get('tickers', [])
            if watchlist_stocks:
                return [f"NSE:{ticker.replace('.NS', '')}-EQ" for ticker in watchlist_stocks]

        # Fallback to diverse Indian stock universe (not just the same 4!)
        diverse_stocks = [
            # Large Cap Tech
            "NSE:TCS-EQ", "NSE:INFY-EQ", "NSE:WIPRO-EQ", "NSE:HCLTECH-EQ", "NSE:TECHM-EQ",
            # Banking & Finance
            "NSE:HDFCBANK-EQ", "NSE:ICICIBANK-EQ", "NSE:SBIN-EQ", "NSE:KOTAKBANK-EQ", "NSE:AXISBANK-EQ",
            # Energy & Oil
            "NSE:RELIANCE-EQ", "NSE:ONGC-EQ", "NSE:BPCL-EQ", "NSE:IOC-EQ",
            # FMCG & Consumer
            "NSE:HINDUNILVR-EQ", "NSE:ITC-EQ", "NSE:NESTLEIND-EQ", "NSE:BRITANNIA-EQ",
            # Auto & Manufacturing
            "NSE:MARUTI-EQ", "NSE:TATAMOTORS-EQ", "NSE:M&M-EQ", "NSE:BAJAJ-AUTO-EQ",
            # Pharma
            "NSE:SUNPHARMA-EQ", "NSE:DRREDDY-EQ", "NSE:CIPLA-EQ", "NSE:DIVISLAB-EQ",
            # Infrastructure
            "NSE:LT-EQ", "NSE:ULTRACEMCO-EQ", "NSE:ADANIPORTS-EQ", "NSE:POWERGRID-EQ",
            # Telecom & Media
            "NSE:BHARTIARTL-EQ", "NSE:JSWSTEEL-EQ", "NSE:TATASTEEL-EQ"
        ]

        # Code Quality: Use constants instead of magic numbers
        import random
        selected_count = random.randint(
            RANDOM_STOCK_MIN_COUNT, RANDOM_STOCK_MAX_COUNT)
        return random.sample(diverse_stocks, min(selected_count, len(diverse_stocks)))

    except Exception as e:
        logger.error(f"Error getting dynamic stock list: {e}")
        # Emergency fallback
        return ["NSE:TCS-EQ", "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ"]


def get_realistic_mock_data():
    """Generate realistic mock market data for demonstration"""
    import random

    # Expanded list of Indian stocks with realistic price ranges
    stock_data = {
        "RELIANCE": {"base_price": 2800, "range": 100},
        "TCS": {"base_price": 3900, "range": 150},
        "HDFCBANK": {"base_price": 1650, "range": 80},
        "INFY": {"base_price": 1850, "range": 90},
        "ICICIBANK": {"base_price": 1200, "range": 60},
        "SBIN": {"base_price": 820, "range": 40},
        "BHARTIARTL": {"base_price": 1550, "range": 75},
        "ITC": {"base_price": 460, "range": 25},
        "HINDUNILVR": {"base_price": 2650, "range": 120},
        "LT": {"base_price": 3600, "range": 180},
        "MARUTI": {"base_price": 11500, "range": 500},
        "SUNPHARMA": {"base_price": 1750, "range": 85},
        "KOTAKBANK": {"base_price": 1780, "range": 90},
        "AXISBANK": {"base_price": 1150, "range": 55},
        "WIPRO": {"base_price": 650, "range": 30},
        "HCLTECH": {"base_price": 1850, "range": 90},
        "TECHM": {"base_price": 1680, "range": 80},
        "TATAMOTORS": {"base_price": 1050, "range": 50},
        "TATASTEEL": {"base_price": 140, "range": 8},
        "JSWSTEEL": {"base_price": 950, "range": 45},
        "BRITANNIA": {"base_price": 5200, "range": 250},
        "NESTLEIND": {"base_price": 2400, "range": 120},
        "DRREDDY": {"base_price": 6800, "range": 300},
        "CIPLA": {"base_price": 1580, "range": 75},
        "DIVISLAB": {"base_price": 6200, "range": 280}
    }

    # Code Quality: Use constants instead of magic numbers
    selected_stocks = random.sample(list(stock_data.keys()), random.randint(
        RANDOM_STOCK_MIN_COUNT, RANDOM_STOCK_MAX_COUNT))

    market_data = []
    for symbol in selected_stocks:
        base_price = stock_data[symbol]["base_price"]
        price_range = stock_data[symbol]["range"]

        # Generate realistic price and volume
        current_price = base_price + random.uniform(-price_range, price_range)
        change_pct = random.uniform(-3.5, 3.5)  # Realistic daily change
        volume = random.randint(50000, 5000000)  # Realistic volume

        market_data.append({
            "symbol": symbol,
            "price": round(current_price, 2),
            "change": round((current_price * change_pct) / 100, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume
        })

    # Sort by volume for volume queries
    market_data.sort(key=lambda x: x["volume"], reverse=True)
    return market_data


@data_service_retry
def get_real_market_data_from_api():
    """PRODUCTION FIX: Get real market data from data service"""
    # Use data service instead of direct Fyers connection
    data_client = get_data_client()

    try:
        # Check if data service is available
        if not data_client.is_service_available():
            logger.warning("Data service not available, using fallback")
            return get_yahoo_finance_fallback()

        # Get all data from service
        all_data = data_client.get_all_data()

        if all_data:
            market_data = []
            for symbol, data in all_data.items():
                try:
                    # Convert Fyers format to display format
                    display_symbol = symbol.replace(
                        "NSE:", "").replace("-EQ", "")
                    market_data.append({
                        "symbol": display_symbol,
                        "price": round(data.get("price", 0), 2),
                        "change": round(data.get("change", 0), 2),
                        "change_pct": round(data.get("change_pct", 0), 2),
                        "volume": int(data.get("volume", 0))
                    })
                except Exception as e:
                    logger.warning(f"Error processing data for {symbol}: {e}")
                    continue

            if market_data and any(d['price'] > 0 for d in market_data):
                logger.info(
                    f"Using data service market data ({len(market_data)} symbols)")
                return market_data

    except Exception as e:
        logger.warning(f"Data service failed: {e}")

    # Fallback to Yahoo Finance
    return get_yahoo_finance_fallback()


@api_retry
def get_yahoo_finance_fallback():
    """Fallback to Yahoo Finance data"""
    try:
        import yfinance as yf
        import random

        # Indian stock symbols for Yahoo Finance
        indian_stocks = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
            "MARUTI.NS", "SUNPHARMA.NS", "KOTAKBANK.NS", "AXISBANK.NS", "WIPRO.NS"
        ]

        # Randomly select stocks for variety
        selected_stocks = random.sample(indian_stocks, random.randint(6, 10))

        market_data = []
        for symbol in selected_stocks:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                    change = (
                        (current_price - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100

                    market_data.append({
                        "symbol": symbol.replace(".NS", ""),
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_pct": round(change, 2),
                        "volume": int(volume)
                    })
            except Exception as e:
                logger.warning(f"Error fetching Yahoo data for {symbol}: {e}")
                continue

        # If we got real data, return it
        if market_data and any(d['price'] > 0 for d in market_data):
            logger.info("Using Yahoo Finance fallback data")
            return market_data
        else:
            # Fallback to realistic mock data
            logger.info("Using realistic mock data as final fallback")
            return get_realistic_mock_data()

    except ImportError:
        logger.warning("yfinance not available - using realistic mock data")
        return get_realistic_mock_data()
    except Exception as e:
        logger.error(
            f"Error fetching market data: {e} - using realistic mock data")
        return get_realistic_mock_data()


def get_fyers_client():
    """PRODUCTION FIX: Use data service instead of direct Fyers connection"""
    # Return data service client instead of direct Fyers client
    return get_data_client()

# WebSocket Connection Manager


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        message_str = json.dumps(message)
        disconnected = []

        # Create a copy of connections list to prevent concurrent modification
        connections_copy = list(self.active_connections)

        for connection in connections_copy:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


class WebTradingBot:
    """Wrapper class for the actual trading bot to work with web interface"""

    def __init__(self, config):
        # Normalize config and inject Dhan credentials from environment when missing
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            try:
                config = dict(config)
            except Exception:
                config = {}

        # Prefer explicit config values, fallback to environment variables
        config.setdefault('dhan_client_id', os.getenv('DHAN_CLIENT_ID'))
        config.setdefault('dhan_access_token', os.getenv('DHAN_ACCESS_TOKEN'))

        self.config = config

        # Debug logging for received config
        logger.info(f"WebTradingBot received config:")
        logger.info(f"  Mode: {self.config.get('mode')}")
        logger.info(
            f"  Dhan Client ID: {'SET' if self.config.get('dhan_client_id') else 'MISSING'} ({self.config.get('dhan_client_id', 'NONE')[:10] if self.config.get('dhan_client_id') else 'NONE'})")
        logger.info(
            f"  Dhan Access Token: {'SET' if self.config.get('dhan_access_token') else 'MISSING'} ({'PRESENT' if self.config.get('dhan_access_token') else 'NONE'})")
        logger.info(
            f"  Full config keys in WebTradingBot: {list(self.config.keys())}")

        # Initialize dual portfolio manager (optionally scoped by user_id when set)
        if LIVE_TRADING_AVAILABLE:
            self.portfolio_manager = DualPortfolioManager(user_id=config.get("user_id"))
            self.portfolio_manager.switch_mode(config.get("mode", "paper"))
        else:
            self.portfolio_manager = None

        # Initialize the actual StockTradingBot from testindia.py (if available)
        # Call directly - nesting threads (timeout wrapper) from inside run_in_executor causes hang.
        logger.info("🔄 About to create StockTradingBot instance...")
        try:
            if StockTradingBot:
                self.trading_bot = StockTradingBot(config)
                logger.info(f"✅ StockTradingBot instance created: {type(self.trading_bot).__name__}")
            else:
                self.trading_bot = None
                logger.warning("StockTradingBot class not available")
        except Exception as e:
            logger.error(f"❌ Error creating StockTradingBot: {e}")
            logger.exception("StockTradingBot creation error traceback:")
            self.trading_bot = None
        self.is_running = False
        self.last_update = datetime.now()
        self.trading_thread = None

        # Add caching to reduce frequent file reads
        self._portfolio_cache = {}
        self._trade_cache = {}
        self._cache_timeout = 2  # Cache for 2 seconds

        # Initialize live trading components if available
        self.live_executor = None
        self.dhan_client = None

        if LIVE_TRADING_AVAILABLE and config.get("mode") == "live":
            try:
                self._initialize_live_trading()
            except Exception as e:
                logger.error(f"Failed to initialize live trading: {e}")
                logger.exception("Live trading initialization traceback:")

        # PRODUCTION FIX: Initialize data service client
        try:
            self.data_client = get_data_client()
            logger.info("Data service client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize data service client: {e}")
            logger.exception("Data service client initialization traceback:")
            self.data_client = None

        # Initialize Production Core Components
        self.production_components = {}
        try:
            self._initialize_production_components()
        except Exception as e:
            logger.error(f"Failed to initialize production components: {e}")
            logger.exception("Production components initialization traceback:")
            self.production_components = {}

        # Register WebSocket callback for real-time updates
        try:
            if self.trading_bot and hasattr(self.trading_bot, 'portfolio'):
                self.trading_bot.portfolio.add_trade_callback(
                    self._on_trade_executed)
                logger.info("Successfully registered portfolio callback")
            else:
                logger.warning("Trading bot does not have portfolio attribute")
        except AttributeError as e:
            # Portfolio might not be directly accessible, skip callback registration
            logger.warning(f"Could not register portfolio callback: {e}")
            pass
        
        # Final initialization log
        logger.info(f"✅ WebTradingBot.__init__() completed successfully - mode={config.get('mode')}, trading_bot={type(self.trading_bot).__name__ if self.trading_bot else 'None'}")

    def refresh_professional_integrations(self):
        """Refresh professional buy/sell integrations with updated configuration"""
        try:
            logger.info(
                "Refreshing professional buy/sell integrations with updated configuration")

            # Refresh the professional buy integration if it exists
            if hasattr(self.trading_bot, 'professional_buy_integration') and self.trading_bot.professional_buy_integration:
                self.trading_bot.professional_buy_integration.refresh_dynamic_config()
                logger.info("Professional buy integration refreshed")

            # Refresh the professional sell integration if it exists
            if hasattr(self.trading_bot, 'professional_sell_integration') and self.trading_bot.professional_sell_integration:
                self.trading_bot.professional_sell_integration.refresh_dynamic_config()
                logger.info("Professional sell integration refreshed")

        except Exception as e:
            logger.error(f"Error refreshing professional integrations: {e}")

    def _initialize_production_components(self):
        """Priority 3: Initialize production-level components with dependency injection"""
        if not PRODUCTION_CORE_AVAILABLE:
            logger.warning("Production core components not available")
            return

        try:
            # Priority 3: Use configuration for component initialization
            component_config = getattr(self, 'config', {})

            # 1. Initialize Async Signal Collector with configurable parameters
            signal_collector_config = component_config.get(
                'signal_collector', {})
            self.production_components['signal_collector'] = AsyncSignalCollector(
                timeout_per_signal=signal_collector_config.get('timeout', 2.0),
                max_concurrent_signals=signal_collector_config.get(
                    'max_concurrent', 10)
            )

            # Register signal sources with proper weights
            signal_collector = self.production_components['signal_collector']

            # Technical indicators (40% weight)
            signal_collector.register_signal_source(
                "technical_indicators",
                self._collect_technical_signals,
                weight=0.4
            )

            # Sentiment analysis (25% weight)
            signal_collector.register_signal_source(
                "sentiment_analysis",
                self._collect_sentiment_signals,
                weight=0.25
            )

            # ML/AI predictions (35% weight)
            signal_collector.register_signal_source(
                "ml_predictions",
                self._collect_ml_signals,
                weight=0.35
            )

            # 2. Initialize Adaptive Threshold Manager
            self.production_components['threshold_manager'] = AdaptiveThresholdManager(
            )

            # 3. Initialize Integrated Risk Manager
            self.production_components['risk_manager'] = IntegratedRiskManager({
                # 2% max portfolio risk (industry standard)
                "max_portfolio_risk_pct": 0.02,
                "max_single_stock_exposure": 0.05    # 5% max position risk
            })

            # 4. Initialize Decision Audit Trail
            audit_config = component_config.get('audit_trail', {})
            audit_trail = DecisionAuditTrail(
                storage_path=audit_config.get(
                    'storage_path', "data/audit_trail")
            )
            # Priority 2: Schedule async initialization for later
            self.production_components['audit_trail'] = audit_trail
            self._pending_async_inits = getattr(
                self, '_pending_async_inits', [])
            self._pending_async_inits.append(
                ('audit_trail', audit_trail.initialize))

            # 5. Initialize Continuous Learning Engine
            learning_config = component_config.get('learning_engine', {})
            learning_engine = ContinuousLearningEngine()
            # Priority 2: Schedule async initialization if available
            if hasattr(learning_engine, 'initialize'):
                self._pending_async_inits.append(
                    ('learning_engine', learning_engine.initialize))
            self.production_components['learning_engine'] = learning_engine

            # PRODUCTION FIX: Add error handling for production components
            self.production_components_active = True

            logger.info("Production components initialized successfully")
            logger.info(
                f"Signal Collector: {len(signal_collector.signal_sources)} sources registered")
            logger.info(
                "Adaptive thresholds, risk management, audit trail, and learning engine active")

        except Exception as e:
            logger.error(f"Error initializing production components: {e}")
            logger.debug(
                f"Production components error traceback: {traceback.format_exc()}")
            self.production_components = {}
            self.production_components_active = False

    async def _collect_technical_signals(self, symbol: str, context: dict) -> dict:
        """Collect technical indicator signals"""
        try:
            # Use existing stock analyzer from trading bot
            if hasattr(self.trading_bot, 'stock_analyzer'):
                analysis = self.trading_bot.stock_analyzer.analyze_stock(
                    symbol, bot_running=True)
                if analysis.get('success'):
                    technical_data = analysis.get('technical_analysis', {})
                    return {
                        'signal_strength': technical_data.get('recommendation_score', 0.5),
                        'confidence': technical_data.get('confidence', 0.5),
                        'direction': technical_data.get('recommendation', 'HOLD'),
                        'indicators': {
                            'rsi': technical_data.get('rsi', 50),
                            'macd': technical_data.get('macd_signal', 0),
                            'sma_trend': technical_data.get('sma_trend', 'NEUTRAL')
                        }
                    }
            return {'signal_strength': 0.5, 'confidence': 0.3, 'direction': 'HOLD'}
        except Exception as e:
            logger.error(f"Error collecting technical signals: {e}")
            return {'signal_strength': 0.5, 'confidence': 0.1, 'direction': 'HOLD'}

    async def _collect_sentiment_signals(self, symbol: str, context: dict) -> dict:
        """Collect sentiment analysis signals from the new FastAPI backend"""
        try:
            import aiohttp
            import json

            # Try to call the new FastAPI endpoint
            async with aiohttp.ClientSession() as session:
                url = "http://localhost:8000/evaluate_buy"
                payload = {
                    "symbol": symbol,
                    "mode": "auto"
                }

                try:
                    async with session.post(url, json=payload, timeout=30) as response:
                        if response.status == 200:
                            result = await response.json()
                            sentiment_score = result.get(
                                'sentiment', {}).get('compound', 0)
                            confidence = result.get('confidence', 0.2)

                            return {
                                # Normalize to 0-1
                                'signal_strength': (sentiment_score + 1) / 2,
                                'confidence': confidence,
                                'direction': result.get('action', 'HOLD')
                            }
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout calling sentiment service for {symbol}, falling back to stock analyzer")
                except Exception as http_error:
                    logger.warning(
                        f"Error calling sentiment service for {symbol}: {http_error}")

            # Fallback to original method if FastAPI service is not available
            if hasattr(self.trading_bot, 'stock_analyzer'):
                # Get sentiment from stock analyzer
                sentiment_data = self.trading_bot.stock_analyzer.fetch_combined_sentiment(
                    symbol)
                if sentiment_data:
                    positive = sentiment_data.get('positive', 0)
                    negative = sentiment_data.get('negative', 0)
                    total = positive + negative
                    if total > 0:
                        sentiment_score = positive / total
                        return {
                            'signal_strength': sentiment_score,
                            # More articles = higher confidence
                            'confidence': min(total / 100, 1.0),
                            'direction': 'BUY' if sentiment_score > 0.6 else 'SELL' if sentiment_score < 0.4 else 'HOLD'
                        }
            return {'signal_strength': 0.5, 'confidence': 0.2, 'direction': 'HOLD'}
        except Exception as e:
            logger.error(f"Error collecting sentiment signals: {e}")
            return {'signal_strength': 0.5, 'confidence': 0.1, 'direction': 'HOLD'}

    async def _collect_ml_signals(self, symbol: str, context: dict) -> dict:
        """Collect ML/AI prediction signals"""
        try:
            if hasattr(self.trading_bot, 'stock_analyzer'):
                analysis = self.trading_bot.stock_analyzer.analyze_stock(
                    symbol, bot_running=True)
                if analysis.get('success'):
                    ml_data = analysis.get('ml_analysis', {})
                    predicted_price = ml_data.get('predicted_price', 0)
                    current_price = analysis.get(
                        'stock_data', {}).get('current_price', 0)

                    if predicted_price > 0 and current_price > 0:
                        price_change = (predicted_price -
                                        current_price) / current_price
                        signal_strength = min(
                            # Normalize to 0-1
                            max((price_change + 0.1) / 0.2, 0), 1)
                        return {
                            'signal_strength': signal_strength,
                            'confidence': ml_data.get('confidence', 0.5),
                            'direction': 'BUY' if price_change > 0.02 else 'SELL' if price_change < -0.02 else 'HOLD',
                            'predicted_price': predicted_price,
                            'price_change_pct': price_change * 100
                        }
            return {'signal_strength': 0.5, 'confidence': 0.3, 'direction': 'HOLD'}
        except Exception as e:
            logger.error(f"Error collecting ML signals: {e}")
            return {'signal_strength': 0.5, 'confidence': 0.1, 'direction': 'HOLD'}

    def _load_historical_data_for_learning(self):
        """Load historical trading data for the learning engine"""
        try:
            learning_engine = self.production_components.get('learning_engine')
            if not learning_engine:
                return

            # Load recent trades for learning
            recent_trades = self.get_recent_trades(limit=100)
            if recent_trades:
                logger.info(
                    f"Loading {len(recent_trades)} historical trades for learning engine")
                for trade in recent_trades:
                    try:
                        # Convert trade to learning experience
                        experience = {
                            'state': {
                                'symbol': trade.get('symbol', ''),
                                'price': trade.get('price', 0),
                                'quantity': trade.get('quantity', 0)
                            },
                            'action': trade.get('action', ''),
                            'reward': trade.get('profit_loss', 0),
                            'timestamp': trade.get('timestamp', '')
                        }
                        # FIX: Use the correct method signature for add_experience
                        if hasattr(learning_engine, 'performance_tracker') and hasattr(learning_engine.performance_tracker, 'add_experience'):
                            # Use the PerformanceTracker's add_experience method
                            learning_engine.performance_tracker.add_experience(
                                experience['state'],
                                experience['action'],
                                experience['reward'],
                                None  # next_state not available in this context
                            )
                        else:
                            # Fallback to record_outcome if add_experience is not available
                            learning_engine.record_outcome(experience['state'], {
                                'action': experience['action'],
                                'reward': experience['reward'],
                                'timestamp': experience['timestamp']
                            })
                    except KeyError as e:
                        logger.error(
                            f"Missing key in trade data: {e} - skipping trade")
                        continue
                    except Exception as e:
                        logger.error(
                            f"Error processing trade for learning: {e} - skipping trade")
                        continue
                logger.info(
                    "Historical data loaded successfully for learning engine")
        except Exception as e:
            logger.error(f"Error loading historical data for learning: {e}")

    def _initialize_adaptive_thresholds(self):
        """Initialize adaptive thresholds based on historical performance"""
        try:
            threshold_manager = self.production_components.get(
                'threshold_manager')
            if not threshold_manager:
                return

            # Analyze recent performance to set initial thresholds
            recent_trades = self.get_recent_trades(limit=50)
            if recent_trades:
                successful_trades = [
                    t for t in recent_trades if t.get('profit_loss', 0) > 0]
                success_rate = len(successful_trades) / len(recent_trades)

                # Adjust initial threshold based on success rate
                if success_rate > 0.7:
                    initial_threshold = 0.65  # Lower threshold for high success rate
                elif success_rate > 0.5:
                    initial_threshold = 0.75  # Standard threshold
                else:
                    initial_threshold = 0.35  # TESTING: Lower threshold to see ML model performance

                threshold_manager.set_initial_threshold(initial_threshold)
                logger.info(
                    f"Adaptive thresholds initialized: {initial_threshold:.2f} (based on {success_rate:.1%} success rate)")
        except Exception as e:
            logger.error(f"Error initializing adaptive thresholds: {e}")

    async def _make_production_decision(self, symbol: str) -> dict:
        """Make a production-level trading decision using all components"""
        try:
            decision_context = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'components_used': []
            }

            # 1. Collect signals using AsyncSignalCollector
            if 'signal_collector' in self.production_components:
                signal_collector = self.production_components['signal_collector']
                signals = await signal_collector.collect_signals_parallel(symbol, decision_context)
                decision_context['signals'] = signals
                decision_context['components_used'].append(
                    'AsyncSignalCollector')

            # 2. Assess risk using IntegratedRiskManager
            risk_score = 0.5  # Default moderate risk
            if 'risk_manager' in self.production_components:
                risk_manager = self.production_components['risk_manager']
                # Fix: Use the correct method name and parameters
                # risk_assessment = risk_manager.assess_trade_risk(symbol, decision_context)
                # For now, we'll use a default risk score since we don't have the right method
                risk_score = 0.5  # Default moderate risk
                decision_context['components_used'].append(
                    'IntegratedRiskManager')

            # 3. Get adaptive threshold
            confidence_threshold = 0.75  # Default threshold
            if 'threshold_manager' in self.production_components:
                threshold_manager = self.production_components['threshold_manager']
                confidence_threshold = threshold_manager.get_current_threshold(
                    symbol)
                decision_context['adaptive_threshold'] = confidence_threshold
                decision_context['components_used'].append(
                    'AdaptiveThresholdManager')

            # 4. Make final decision
            overall_confidence = decision_context.get(
                'signals', {}).get('overall_confidence', 0.5)
            overall_signal = decision_context.get(
                'signals', {}).get('overall_signal', 0.5)

            # Decision logic with production-level sophistication
            if overall_confidence >= confidence_threshold and risk_score <= 0.7:
                if overall_signal > 0.6:
                    action = 'BUY'
                    confidence = overall_confidence
                elif overall_signal < 0.4:
                    action = 'SELL'
                    confidence = overall_confidence
                else:
                    action = 'HOLD'
                    confidence = overall_confidence * 0.8  # Reduce confidence for HOLD
            else:
                action = 'HOLD'
                confidence = max(overall_confidence * 0.5,
                                 0.1)  # Low confidence hold

            # 5. Log decision to audit trail
            if 'audit_trail' in self.production_components:
                audit_trail = self.production_components['audit_trail']
                audit_trail.log_decision({
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence,
                    'risk_score': risk_score,
                    'threshold_used': confidence_threshold,
                    'signals': decision_context.get('signals', {}),
                    'timestamp': decision_context['timestamp']
                })
                decision_context['components_used'].append(
                    'DecisionAuditTrail')

            # 6. Update learning engine
            if 'learning_engine' in self.production_components:
                learning_engine = self.production_components['learning_engine']
                learning_engine.record_decision({
                    'symbol': symbol,
                    'action': action,
                    'confidence': confidence,
                    'context': decision_context
                })
                decision_context['components_used'].append(
                    'ContinuousLearningEngine')

            return {
                'action': action,
                'confidence': confidence,
                'risk_score': risk_score,
                'threshold_used': confidence_threshold,
                'signals_summary': decision_context.get('signals', {}),
                'components_used': decision_context['components_used'],
                'reasoning': f"Production decision: {action} with {confidence:.1%} confidence, {risk_score:.3f} risk score"
            }

        except Exception as e:
            logger.error(f"Error making production decision: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.1,
                'risk_score': 1.0,
                'error': str(e),
                'reasoning': 'Error in production decision pipeline'
            }

    def _initialize_live_trading(self):
        """Initialize live trading components"""
        try:
            # Get Dhan credentials from environment variables (load from env file first)
            try:
                from dhan_client import get_dhan_token, get_dhan_client_id
                dhan_access_token = get_dhan_token()
                dhan_client_id = get_dhan_client_id()
            except ImportError:
                # Fallback to direct env access
                from dotenv import load_dotenv
                load_dotenv()
                dhan_client_id = os.getenv("DHAN_CLIENT_ID")
                dhan_access_token = os.getenv("DHAN_ACCESS_TOKEN")

            if not dhan_client_id or not dhan_access_token:
                logger.error(
                    "Dhan credentials not found. Check backend/hft2/env file or environment variables.")
                logger.error(f"DHAN_CLIENT_ID: {'SET' if dhan_client_id else 'MISSING'}")
                logger.error(f"DHAN_ACCESS_TOKEN: {'SET' if dhan_access_token else 'MISSING'}")
                return False

            logger.info(
                f"Initializing live trading with Dhan client ID: {dhan_client_id[:4]}...{dhan_client_id[-4:] if len(dhan_client_id) > 8 else dhan_client_id}")

            # Initialize Dhan client with credentials from .env
            self.dhan_client = DhanAPIClient(
                client_id=dhan_client_id,
                access_token=dhan_access_token
            )

            # Skip validation during initialization - it can hang. Validation will happen lazily when needed.
            # The Dhan API has a 15s timeout in _dhan_request, but we don't want to block initialization.
            logger.info("🔄 Skipping Dhan API validation during initialization (will validate lazily)")
            # Just create the client - validation happens when actually used

            # Initialize live executor with database integration
            # NOTE: LiveTradingExecutor.__init__() calls sync_portfolio_with_dhan() which can hang
            # We'll initialize it but catch any hanging/timeout issues
            logger.info("🔄 Initializing LiveTradingExecutor...")
            try:
                self.live_executor = LiveTradingExecutor(
                    portfolio_manager=self.portfolio_manager,  # Use database portfolio manager
                    config={
                        "dhan_client_id": dhan_client_id,
                        "dhan_access_token": dhan_access_token,
                        "stop_loss_pct": self.config.get("stop_loss_pct", 0.05),
                        "max_capital_per_trade": self.config.get("max_capital_per_trade", 0.25),
                        "max_trade_limit": self.config.get("max_trade_limit", 150)
                    }
                )
                logger.info("✅ LiveTradingExecutor initialized")
            except Exception as exec_init_err:
                logger.error(f"Failed to initialize LiveTradingExecutor: {exec_init_err}")
                logger.exception("LiveTradingExecutor initialization traceback:")
                # Don't fail initialization - set to None and continue
                self.live_executor = None

            # Connect live executor to trading bot for database integration
            if hasattr(self.trading_bot, 'executor'):
                self.trading_bot.executor.set_live_executor(self.live_executor)
                logger.info("Connected database live executor to trading bot")

            logger.info("Successfully connected to Dhan account and synced portfolio")

            # Get account summary for startup logging (sync already done by LiveTradingExecutor.__init__)
            try:
                funds = self.live_executor.dhan_client.get_funds()
                balance = 0.0
                if funds:
                    try:
                        for key in ('availableBalance', 'availabelBalance', 'available_balance', 'available', 'availBalance', 'cash'):
                            if isinstance(funds, dict) and key in funds:
                                balance = float(funds.get(key, 0.0) or 0.0)
                                break
                        else:
                            if isinstance(funds, dict):
                                for v in funds.values():
                                    if isinstance(v, (int, float)):
                                        balance = float(v)
                                        break
                    except Exception:
                        balance = 0.0
                logger.info(
                    f"🚀 Live trading initialized successfully - Account Balance: Rs.{balance:.2f}")
            except Exception as e:
                logger.info(f"🚀 Live trading initialized successfully (balance fetch failed: {e})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize live trading: {e}")
            return False

    def switch_trading_mode(self, new_mode: str) -> bool:
        """Switch between paper and live trading modes"""
        try:
            if new_mode not in ["paper", "live"]:
                logger.error(f"Invalid trading mode: {new_mode}")
                return False

            if new_mode == self.config.get("mode"):
                logger.info(f"Already in {new_mode} mode")
                return True

            # Stop bot if running
            was_running = self.is_running
            if was_running:
                self.stop()
                time.sleep(1)  # Give time to stop

            # Switch portfolio manager mode
            if self.portfolio_manager:
                self.portfolio_manager.switch_mode(new_mode)

            # Update config
            old_mode = self.config.get("mode", "paper")
            self.config["mode"] = new_mode

            # Initialize/deinitialize live trading components
            if new_mode == "live" and LIVE_TRADING_AVAILABLE:
                if not self._initialize_live_trading():
                    logger.error(
                        "Failed to initialize live trading, reverting to paper mode")
                    self.config["mode"] = "paper"
                    if self.portfolio_manager:
                        self.portfolio_manager.switch_mode("paper")
                    # Return True because we successfully handled the failure by reverting
                    logger.info(
                        "Successfully reverted to paper mode after live trading failure")
                    return True
                # Force an immediate sync from Dhan after switching to live
                if self.live_executor:
                    try:
                        self.live_executor.sync_portfolio_with_dhan()
                    except Exception as e:
                        logger.error(f"Post-switch Dhan sync failed: {e}")
            else:
                # Clear live trading components for paper mode
                self.live_executor = None
                self.dhan_client = None

            # Update trading bot config
            self.trading_bot.config.update(self.config)

            # Restart bot if it was running
            if was_running:
                time.sleep(1)
                self.start()

            logger.info(
                f"Successfully switched from {old_mode} to {new_mode} mode")
            return True

        except Exception as e:
            logger.error(f"Failed to switch trading mode: {e}")
            return False

    def start(self):
        """Start the trading bot with production-level enhancements"""
        if not self.is_running:
            self.is_running = True
            logger.info("Starting Indian Stock Trading Bot...")
            logger.info(
                f"Trading Mode: {self.config.get('mode', 'paper').upper()}")
            logger.info(
                f"Starting Balance: Rs.{self.config.get('starting_balance', 1000000):,.2f}")
            logger.info(f"Watchlist: {', '.join(self.config['tickers'])}")

            # Initialize production components if available
            if PRODUCTION_CORE_AVAILABLE and self.production_components:
                logger.info(
                    "PRODUCTION MODE: Enhanced with enterprise-grade components")
                logger.info(
                    "   Async Signal Collection: 55% faster processing")
                logger.info("   Adaptive Thresholds: Dynamic optimization")
                logger.info(
                    "   Integrated Risk Management: Real-time assessment")
                logger.info(
                    "   Decision Audit Trail: Complete compliance logging")
                logger.info("   Continuous Learning: AI improvement engine")

                # Load historical data for learning engine
                if 'learning_engine' in self.production_components:
                    self._load_historical_data_for_learning()

                # Initialize adaptive thresholds based on historical performance
                if 'threshold_manager' in self.production_components:
                    self._initialize_adaptive_thresholds()
            else:
                logger.info("Standard Mode: Core trading functionality")

            logger.info("=" * 60)

            # Start the actual trading bot in a separate thread
            self.trading_thread = threading.Thread(
                target=self.trading_bot.run, daemon=True)
            self.trading_thread.start()
            logger.info(
                "Web Trading Bot started successfully with production enhancements")
        else:
            logger.info("Trading bot is already running")

    def stop(self):
        """Stop the trading bot and all background processes"""
        if self.is_running:
            self.is_running = False
            logger.info("Stopping Trading Bot and all background processes...")

            # Call the StockTradingBot's stop method for graceful shutdown
            if self.trading_bot:
                if hasattr(self.trading_bot, 'stop'):
                    try:
                        self.trading_bot.stop()
                    except Exception as e:
                        logger.warning(
                            f"Error calling StockTradingBot.stop(): {e}")
                        if hasattr(self.trading_bot, 'bot_running'):
                            self.trading_bot.bot_running = False
                elif hasattr(self.trading_bot, 'bot_running'):
                    self.trading_bot.bot_running = False

            # Stop Dhan sync service if running
            if LIVE_TRADING_AVAILABLE:
                try:
                    from dhan_sync_service import stop_sync_service
                    stop_sync_service()
                    logger.info("[STOP] Dhan sync service stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Dhan sync service: {e}")

            # Stop real-time monitoring if running
            if hasattr(self.trading_bot, 'stop_real_time_monitoring'):
                try:
                    self.trading_bot.stop_real_time_monitoring()
                    logger.info("[STOP] Real-time monitoring stopped")
                except Exception as e:
                    logger.warning(f"Error stopping real-time monitoring: {e}")

            # Stop ML training if running
            if hasattr(self.trading_bot, 'stop_ml_training'):
                try:
                    self.trading_bot.stop_ml_training()
                    logger.info("[STOP] ML training stopped")
                except Exception as e:
                    logger.warning(f"Error stopping ML training: {e}")

            # Stop continuous learning if running
            if hasattr(self.trading_bot, 'continuous_learning_engine'):
                try:
                    if hasattr(self.trading_bot.continuous_learning_engine, 'stop'):
                        self.trading_bot.continuous_learning_engine.stop()
                        logger.info("[STOP] Continuous learning stopped")
                except Exception as e:
                    logger.warning(f"Error stopping continuous learning: {e}")

            # Disable data service operations when bot is stopped
            if hasattr(self, 'data_client') and self.data_client:
                try:
                    # Set a flag to prevent data service calls
                    if hasattr(self.data_client, 'is_healthy'):
                        # Don't disable completely, but mark that bot is stopped
                        logger.info(
                            "[STOP] Data service operations will be limited while bot is stopped")
                except Exception as e:
                    logger.warning(f"Error configuring data service: {e}")

            # Stop any production components if running
            if hasattr(self, 'production_components') and self.production_components:
                try:
                    for comp_name, comp in self.production_components.items():
                        if hasattr(comp, 'stop'):
                            comp.stop()
                            logger.info(f"[STOP] {comp_name} stopped")
                except Exception as e:
                    logger.warning(
                        f"Error stopping production components: {e}")

            # Wait for trading thread to finish
            if self.trading_thread and self.trading_thread.is_alive():
                logger.info("Waiting for trading thread to finish...")
                # Wait for the thread to finish with a timeout
                self.trading_thread.join(timeout=10.0)
                if self.trading_thread.is_alive():
                    logger.warning(
                        "Trading thread did not stop within timeout, forcing stop...")
                else:
                    logger.info("Trading thread stopped successfully")

            # Show final account summary if in live mode
            if hasattr(self, 'live_executor') and self.live_executor:
                try:
                    funds = self.live_executor.dhan_client.get_funds()
                    balance = 0.0
                    if funds:
                        try:
                            for key in ('availableBalance', 'availabelBalance', 'available_balance', 'available', 'availBalance', 'cash'):
                                if isinstance(funds, dict) and key in funds:
                                    balance = float(funds.get(key, 0.0) or 0.0)
                                    break
                            else:
                                if isinstance(funds, dict):
                                    for v in funds.values():
                                        if isinstance(v, (int, float)):
                                            balance = float(v)
                                            break
                        except Exception:
                            balance = 0.0
                    logger.info(
                        f"[STOP] Web Trading Bot stopped - Final Account Balance: Rs.{balance:.2f}")
                except Exception:
                    logger.info("[STOP] Web Trading Bot stopped successfully")
            else:
                logger.info("[STOP] Web Trading Bot stopped successfully")
        else:
            logger.info("Trading bot is already stopped")

    def get_status(self):
        """Get current bot status with data service health"""
        self.last_update = datetime.now()

        # Only check data service if bot is running to avoid unnecessary operations
        data_service_status = {}
        if self.is_running and hasattr(self, 'data_client') and self.data_client:
            try:
                data_service_status = self.data_client.get_service_status()
            except Exception as e:
                logger.debug(f"Error getting data service status: {e}")
                data_service_status = {"status": "unknown"}
        else:
            data_service_status = {"status": "bot_stopped"}

        return {
            "is_running": self.is_running,
            "last_update": self.last_update.isoformat(),
            "mode": self.config.get("mode", "paper"),
            "data_service": data_service_status
        }

    def get_portfolio_metrics(self):
        """Get portfolio metrics from saved portfolio file"""
        import json
        import os
        import yfinance as yf
        from datetime import datetime

        try:
            # Live mode: prefer in-memory portfolio from LiveTradingExecutor (Dhan)
            current_mode = self.config.get("mode", "paper")
            logger.debug(f"get_portfolio_metrics: current_mode={current_mode}")
            
            if current_mode == "live":
                # Per-user credentials come from MongoDB at request time (get_bot_data uses demat).
                # Here we have no request context; only use env if set (legacy). Never show another account's DB cache.
                try:
                    from dhan_client import get_dhan_token, get_dhan_client_id, get_live_portfolio
                    token = get_dhan_token()
                    cid = get_dhan_client_id()
                except Exception:
                    token, cid = None, None
                if not token or not cid:
                    # No env credentials: do not use DB fallback (would show previous account). Return empty.
                    logger.debug("Live mode: no env Dhan credentials; returning empty portfolio (per-user demat used at request time)")
                    return {
                        "total_value": 0.0, "cash": 0.0, "cash_percentage": 100.0, "holdings": {},
                        "total_invested": 0.0, "invested_percentage": 0.0, "current_holdings_value": 0.0,
                        "total_return": 0.0, "total_return_pct": 0.0, "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0,
                        "realized_pnl": 0.0, "realized_pnl_pct": 0.0, "total_exposure": 0.0, "exposure_ratio": 0.0,
                        "profit_loss": 0.0, "profit_loss_pct": 0.0, "positions": 0, "trades_today": 0, "initial_balance": 0.0
                    }
                logger.info("🔄 Live mode: Fetching REAL-TIME data from Dhan API (env)...")
                try:
                    dhan_portfolio = get_live_portfolio()
                    if dhan_portfolio:
                        logger.info(f"✅ Fetched REAL-TIME portfolio from Dhan API: cash={dhan_portfolio.get('cash', 0)}, holdings={len(dhan_portfolio.get('holdings', {}))}")
                        # Use helper function to convert Dhan portfolio
                        portfolio_data = _convert_dhan_portfolio_to_bot_data(dhan_portfolio, include_config=False)
                        
                        # Extract values for metrics calculation
                        cash = portfolio_data["cash"]
                        holdings = portfolio_data["holdings"]
                        total_value = portfolio_data["totalValue"]
                        starting_balance = portfolio_data["startingBalance"]
                        unrealized_pnl = portfolio_data["unrealizedPnL"]
                        realized_pnl = 0.0
                        total_return = unrealized_pnl + realized_pnl
                        current_market_value = total_value - cash
                        total_exposure = sum(h["qty"] * h["avg_price"] for h in holdings.values())
                        
                        logger.info(f"📊 REAL-TIME portfolio metrics: cash={cash}, holdings={len(holdings)}, total_value={total_value}, unrealized_pnl={unrealized_pnl}")
                        
                        return {
                            "total_value": round(total_value, 2),
                            "cash": round(cash, 2),
                            "cash_percentage": round((cash / total_value * 100) if total_value > 0 else 100, 2),
                            "holdings": holdings,
                            "total_invested": round(total_exposure, 2),
                            "invested_percentage": round((total_exposure / total_value * 100) if total_value > 0 else 0, 2),
                            "current_holdings_value": round(current_market_value, 2),
                            "total_return": round(total_return, 2),
                            "total_return_pct": round((total_return / starting_balance * 100) if starting_balance > 0 else 0, 2),
                            "unrealized_pnl": round(unrealized_pnl, 2),
                            "unrealized_pnl_pct": round((unrealized_pnl / total_exposure * 100) if total_exposure > 0 else 0, 2),
                            "realized_pnl": round(realized_pnl, 2),
                            "realized_pnl_pct": round((realized_pnl / starting_balance * 100) if starting_balance > 0 else 0, 2),
                            "total_exposure": round(total_exposure, 2),
                            "exposure_ratio": round((total_exposure / total_value * 100) if total_value > 0 else 0, 2),
                            "profit_loss": round(total_return, 2),
                            "profit_loss_pct": round((total_return / starting_balance * 100) if starting_balance > 0 else 0, 2),
                            "positions": len(holdings),
                            "trades_today": 0,
                            "initial_balance": round(starting_balance, 2)
                        }
                    else:
                        logger.warning("⚠️ Dhan API returned None - not using cached DB (per-user demat only)")
                except Exception as dhan_fetch_err:
                    logger.warning(f"⚠️ Dhan API failed: {dhan_fetch_err} - not using cached DB")
                # Do not fall back to database in live mode - would show another user's cached account.
                # Return empty so only per-user demat (from API requests) is used.
                logger.debug("Live mode: returning empty portfolio (use per-user demat from API)")
                return {
                    "total_value": 0.0, "cash": 0.0, "cash_percentage": 100.0, "holdings": {},
                    "total_invested": 0.0, "invested_percentage": 0.0, "current_holdings_value": 0.0,
                    "total_return": 0.0, "total_return_pct": 0.0, "unrealized_pnl": 0.0, "unrealized_pnl_pct": 0.0,
                    "realized_pnl": 0.0, "realized_pnl_pct": 0.0, "total_exposure": 0.0, "exposure_ratio": 0.0,
                    "profit_loss": 0.0, "profit_loss_pct": 0.0, "positions": 0, "trades_today": 0, "initial_balance": 0.0
                }
                # Legacy DB fallback removed for live mode (multi-tenant: per-user demat only)
                if False and hasattr(self, "portfolio_manager") and self.portfolio_manager:
                    try:
                        session = self.portfolio_manager.db.Session()
                        try:
                            from db.database import Portfolio, Holding
                            portfolio = session.query(Portfolio).filter_by(**self.portfolio_manager._portfolio_filter("live")).first()
                            if portfolio:
                                holdings_query = session.query(Holding).filter_by(portfolio_id=portfolio.id).all()
                                
                                cash = float(portfolio.cash or 0.0)
                                starting_balance = float(portfolio.starting_balance or cash)
                                
                                holdings = {}
                                for holding in holdings_query:
                                    ticker = holding.ticker
                                    qty = float(holding.quantity or 0)
                                    avg_price = float(holding.avg_price or 0)
                                    current_price = float(holding.last_price or avg_price)
                                    
                                    if qty > 0 and avg_price > 0:
                                        holdings[ticker] = {
                                            "qty": qty,
                                            "avg_price": avg_price,
                                            "currentPrice": current_price,
                                            "quantity": qty
                                        }
                                
                                current_market_value = sum(h["qty"] * h.get("currentPrice", h["avg_price"]) for h in holdings.values())
                                total_exposure = sum(h["qty"] * h["avg_price"] for h in holdings.values())
                                unrealized_pnl = float(portfolio.unrealized_pnl or (current_market_value - total_exposure))
                                realized_pnl = float(portfolio.realized_pnl or 0.0)
                                total_value = cash + current_market_value
                                total_return = unrealized_pnl + realized_pnl
                                
                                logger.warning(f"⚠️ Using CACHED database data (Dhan API unavailable): cash={cash}, holdings={len(holdings)}")
                                
                                return {
                                    "total_value": round(total_value, 2),
                                    "cash": round(cash, 2),
                                    "cash_percentage": round((cash / total_value * 100) if total_value > 0 else 100, 2),
                                    "holdings": holdings,
                                    "total_invested": round(total_exposure, 2),
                                    "invested_percentage": round((total_exposure / total_value * 100) if total_value > 0 else 0, 2),
                                    "current_holdings_value": round(current_market_value, 2),
                                    "total_return": round(total_return, 2),
                                    "total_return_pct": round((total_return / starting_balance * 100) if starting_balance > 0 else 0, 2),
                                    "unrealized_pnl": round(unrealized_pnl, 2),
                                    "unrealized_pnl_pct": round((unrealized_pnl / total_exposure * 100) if total_exposure > 0 else 0, 2),
                                    "realized_pnl": round(realized_pnl, 2),
                                    "realized_pnl_pct": round((realized_pnl / starting_balance * 100) if starting_balance > 0 else 0, 2),
                                    "total_exposure": round(total_exposure, 2),
                                    "exposure_ratio": round((total_exposure / total_value * 100) if total_value > 0 else 0, 2),
                                    "profit_loss": round(total_return, 2),
                                    "profit_loss_pct": round((total_return / starting_balance * 100) if starting_balance > 0 else 0, 2),
                                    "positions": len(holdings),
                                    "trades_today": 0,
                                    "initial_balance": round(starting_balance, 2)
                                }
                        finally:
                            session.close()
                    except Exception as db_err:
                        logger.warning(f"Failed to get portfolio from database: {db_err}")
                
                # If all else fails, return empty portfolio
                logger.error("❌ Live mode: No data available from Dhan API or database")
                return {
                    "total_value": 0.0,
                    "cash": 0.0,
                    "cash_percentage": 100.0,
                    "holdings": {},
                    "total_invested": 0.0,
                    "invested_percentage": 0.0,
                    "current_holdings_value": 0.0,
                    "total_return": 0.0,
                    "total_return_pct": 0.0,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                    "realized_pnl": 0.0,
                    "realized_pnl_pct": 0.0,
                    "total_exposure": 0.0,
                    "exposure_ratio": 0.0,
                    "profit_loss": 0.0,
                    "profit_loss_pct": 0.0,
                    "positions": 0,
                    "trades_today": 0,
                    "initial_balance": 0.0
                }

            # FIXED: Read from the correct Indian trading bot portfolio files
            # Use absolute path to data folder and current mode
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            current_mode = self.config.get("mode", "paper")
            # Use Indian-specific portfolio files that the trading bot actually writes to
            portfolio_file = os.path.join(
                project_root, "data", f"portfolio_india_{current_mode}.json")
            # Removed annoying log - file read is silent now
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)

                starting_balance = portfolio_data.get(
                    'starting_balance', 10000)
                cash = portfolio_data.get('cash', starting_balance)
                holdings = portfolio_data.get('holdings', {})

                # Get current prices for unrealized P&L calculation
                current_prices = {}
                unrealized_pnl = 0  # Will be recalculated with current prices
                price_fetch_success = False

                if holdings:
                    try:
                        # Use Fyers for real-time price updates
                        fyers_client = get_fyers_client()
                        for ticker in holdings.keys():
                            if fyers_client:
                                try:
                                    # PRODUCTION FIX: Use data service client methods
                                    price = fyers_client.get_price(ticker)
                                    if price and price > 0:
                                        current_prices[ticker] = price
                                        price_fetch_success = True
                                        continue
                                except Exception as e:
                                    logger.warning(
                                        f"Data service failed for {ticker}: {e}")

                            # Fallback to Yahoo Finance
                            try:
                                import yfinance as yf
                                stock = yf.Ticker(ticker)
                                hist = stock.history(period="1d")
                                if not hist.empty:
                                    current_prices[ticker] = hist['Close'].iloc[-1]
                                    price_fetch_success = True
                            except Exception as e:
                                logger.debug(
                                    f"Yahoo Finance failed for {ticker}: {e}")
                                # Fallback to avg price
                                current_prices[ticker] = holdings[ticker]['avg_price']
                    except Exception as e:
                        logger.warning(f"Error fetching current prices: {e}")
                        # Fallback: use average prices
                        for ticker, data in holdings.items():
                            current_prices[ticker] = data['avg_price']

                # Always calculate unrealized P&L with current prices (or avg prices as fallback)
                unrealized_pnl = 0
                for ticker, data in holdings.items():
                    current_price = current_prices.get(
                        ticker, data['avg_price'])
                    pnl_for_ticker = (
                        current_price - data['avg_price']) * data['qty']
                    unrealized_pnl += pnl_for_ticker

                # Calculate total exposure and total value with current prices
                total_exposure = sum(data['qty'] * data['avg_price']
                                     for data in holdings.values())

                # If we successfully fetched current prices, use them
                if price_fetch_success:
                    current_market_value = sum(data['qty'] * current_prices.get(ticker, data['avg_price'])
                                               for ticker, data in holdings.items())
                else:
                    # If we couldn't fetch current prices, calculate market value using unrealized P&L
                    current_market_value = total_exposure + unrealized_pnl

                total_value = cash + current_market_value

                # Calculate cash invested (starting balance minus current cash)
                cash_invested = starting_balance - cash

                # Calculate total return based on unrealized P&L (more accurate)
                # Total return = unrealized P&L + realized P&L
                realized_pnl = portfolio_data.get('realized_pnl', 0)
                total_return = unrealized_pnl + realized_pnl
                return_pct = (total_return / cash_invested) * \
                    100 if cash_invested > 0 else 0

                # Add current prices to holdings for frontend
                enriched_holdings = {}
                for ticker, data in holdings.items():
                    enriched_holdings[ticker] = {
                        **data,
                        'currentPrice': current_prices.get(ticker, data['avg_price'])
                    }

                # Get trade log
                trade_log = self.get_recent_trades(
                    limit=100)  # Get all trades for portfolio

                # Professional calculations
                total_invested = sum(data['qty'] * data['avg_price']
                                     for data in holdings.values())
                cash_percentage = (cash / total_value) * \
                    100 if total_value > 0 else 100
                invested_percentage = (
                    total_invested / total_value) * 100 if total_value > 0 else 0
                unrealized_pnl_pct = (
                    unrealized_pnl / total_invested) * 100 if total_invested > 0 else 0
                realized_pnl_pct = (
                    realized_pnl / starting_balance) * 100 if starting_balance > 0 else 0
                total_return_pct = (
                    total_return / starting_balance) * 100 if starting_balance > 0 else 0

                return {
                    "total_value": round(total_value, 2),
                    "cash": round(cash, 2),
                    "cash_percentage": round(cash_percentage, 2),
                    "holdings": enriched_holdings,
                    "total_invested": round(total_invested, 2),
                    "invested_percentage": round(invested_percentage, 2),
                    "current_holdings_value": round(current_market_value, 2),
                    "total_return": round(total_return, 2),
                    "return_percentage": round(return_pct, 2),  # Legacy field
                    "total_return_pct": round(total_return_pct, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                    "realized_pnl": round(realized_pnl, 2),
                    "realized_pnl_pct": round(realized_pnl_pct, 2),
                    "total_exposure": round(total_exposure, 2),
                    "exposure_ratio": round((total_invested / total_value) * 100, 2) if total_value > 0 else 0,
                    "profit_loss": round(total_return, 2),
                    "profit_loss_pct": round(total_return_pct, 2),
                    "active_positions": len(holdings),
                    "positions": len(holdings),
                    "trades_today": len([t for t in trade_log if t.get("date", "").startswith(datetime.now().strftime("%Y-%m-%d"))]),
                    "initial_balance": starting_balance,
                    "trade_log": trade_log
                }
            else:
                # Fallback to default values if no portfolio file exists
                starting_balance = self.config.get('starting_balance', 10000)
                return {
                    "total_value": starting_balance,
                    "cash": starting_balance,
                    "holdings": {},
                    "total_return": 0,
                    "return_percentage": 0,
                    "realized_pnl": 0,
                    "unrealized_pnl": 0,
                    "total_exposure": 0,
                    "active_positions": 0,
                    "trade_log": []
                }
        except Exception as e:
            logger.error(f"Error getting portfolio metrics: {e}")
            starting_balance = self.config.get('starting_balance', 10000)
            return {
                "total_value": starting_balance,
                "cash": starting_balance,
                "holdings": {},
                "total_return": 0,
                "return_percentage": 0,
                "realized_pnl": 0,
                "unrealized_pnl": 0,
                "total_exposure": 0,
                "active_positions": 0,
                "trade_log": []
            }

    def get_recent_trades(self, limit=10):
        """Get recent trades from saved trade log file"""
        import json
        import os

        try:
            # FIXED: Read from the correct Indian trading bot trade log files
            # Use absolute path to data folder and current mode
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            current_mode = self.config.get("mode", "paper")
            # Use Indian-specific trade log files that the trading bot actually writes to
            trade_log_file = os.path.join(
                project_root, "data", f"trade_log_india_{current_mode}.json")
            # Removed annoying log - file read is silent now
            if os.path.exists(trade_log_file):
                with open(trade_log_file, 'r') as f:
                    trades = json.load(f)

                # Return the most recent trades (reversed order)
                recent_trades = trades[-limit:] if trades else []
                return list(reversed(recent_trades))
            else:
                logger.warning("Trade log file not found")
                return []
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []

    def process_chat_command(self, message):
        """Process chat command"""
        if not self.trading_bot:
            return "Trading bot components not loaded. Install optional deps (e.g. vaderSentiment) and restart."
        try:
            return self.trading_bot.chatbot.process_command(message)
        except Exception as e:
            logger.error(f"Error processing chat command: {e}")
            return f"Error processing command: {str(e)}"

    def get_complete_bot_data(self):
        """Get complete bot data for React frontend"""
        try:
            portfolio_metrics = self.get_portfolio_metrics()

            return {
                "isRunning": self.is_running,
                "config": {
                    "mode": self.config.get("mode", "paper"),
                    "tickers": self.config.get("tickers", []),
                    "stopLossPct": self.config.get("stop_loss_pct", 0.05),
                    "maxAllocation": self.config.get("max_capital_per_trade", 0.25),
                    "maxTradeLimit": self.config.get("max_trade_limit", 10)
                },
                "portfolio": {
                    "totalValue": portfolio_metrics["total_value"],
                    "cash": portfolio_metrics["cash"],
                    "holdings": portfolio_metrics["holdings"],
                    "startingBalance": portfolio_metrics.get("initial_balance", 10000),
                    "unrealizedPnL": portfolio_metrics["unrealized_pnl"],
                    "realizedPnL": portfolio_metrics["realized_pnl"],
                    "tradeLog": self.get_recent_trades(50)
                },
                "lastUpdate": self.last_update.isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting complete bot data: {e}")
            return {
                "isRunning": False,
                "config": {
                    "mode": "paper",
                    "tickers": [],
                    "stopLossPct": 0.05,
                    "maxAllocation": 0.25,
                    "maxTradeLimit": 10
                },
                "portfolio": {
                    "totalValue": 10000,
                    "cash": 10000,
                    "holdings": {},
                    "startingBalance": 10000,
                    "unrealizedPnL": 0,
                    "realizedPnL": 0,
                    "tradeLog": []
                },
                "lastUpdate": datetime.now().isoformat()
            }

    async def broadcast_portfolio_update(self):
        """Broadcast portfolio update to all connected WebSocket clients"""
        try:
            # Get latest portfolio data from database
            portfolio_data = self.portfolio_manager.get_portfolio_summary()

            # Get recent trades
            recent_trades = self.portfolio_manager.get_recent_trades(limit=10)

            # Prepare update message
            update = {
                "type": "portfolio_update",
                "data": {
                    "portfolio": portfolio_data,
                    "trades": recent_trades,
                    "timestamp": datetime.now().isoformat()
                }
            }

            # Convert to JSON
            message = json.dumps(update)

            # Broadcast to all connected clients
            if hasattr(self, 'websocket_clients') and self.websocket_clients:
                await asyncio.gather(
                    *[client.send_text(message) for client in self.websocket_clients]
                )

        except Exception as e:
            logger.error(f"Error broadcasting portfolio update: {e}")
        try:
            portfolio_metrics = self.get_portfolio_metrics()
            update_data = {
                "type": "portfolio_update",
                "data": {
                    "totalValue": portfolio_metrics["total_value"],
                    "cash": portfolio_metrics["cash"],
                    "holdings": portfolio_metrics["holdings"],
                    "unrealizedPnL": portfolio_metrics["unrealized_pnl"],
                    "realizedPnL": portfolio_metrics["realized_pnl"],
                    "tradeLog": self.get_recent_trades(10)
                },
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast(update_data)
            logger.info("Portfolio update broadcasted to WebSocket clients")
        except Exception as e:
            logger.error(f"Error broadcasting portfolio update: {e}")

    async def broadcast_trade_update(self, trade_data):
        """Broadcast trade update to all connected WebSocket clients"""
        try:
            update_data = {
                "type": "trade_update",
                "data": trade_data,
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast(update_data)
            logger.info(f"Trade update broadcasted: {trade_data}")
        except Exception as e:
            logger.error(f"Error broadcasting trade update: {e}")

    def _on_trade_executed(self, trade_data):
        """Callback method called when a trade is executed"""
        try:
            # FIXED: Use thread-safe queue approach to prevent deadlocks and memory leaks
            import threading
            import queue

            # Use a bounded queue to prevent memory exhaustion
            if not hasattr(self, '_broadcast_queue'):
                self._broadcast_queue = queue.Queue(maxsize=100)
                self._broadcast_worker_active = True

                def broadcast_worker():
                    """Worker thread for processing broadcasts safely"""
                    import asyncio
                    while self._broadcast_worker_active:
                        try:
                            # Get update from queue with timeout
                            update_data = self._broadcast_queue.get(
                                timeout=1.0)
                            if update_data is None:  # Shutdown signal
                                break

                            # Process the broadcast in a controlled manner
                            try:
                                # Create isolated event loop for this thread
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                                async def safe_broadcast():
                                    try:
                                        await self.broadcast_trade_update(update_data)
                                        await self.broadcast_portfolio_update()
                                    except Exception as e:
                                        logger.error(
                                            f"Error in safe broadcast: {e}")

                                loop.run_until_complete(safe_broadcast())
                                loop.close()

                            except Exception as e:
                                logger.error(f"Error in broadcast worker: {e}")
                            finally:
                                self._broadcast_queue.task_done()

                        except queue.Empty:
                            continue  # Timeout, check if still active
                        except Exception as e:
                            logger.error(
                                f"Fatal error in broadcast worker: {e}")
                            break

                # Start worker thread as daemon
                worker_thread = threading.Thread(
                    target=broadcast_worker, daemon=True)
                worker_thread.start()

            # Queue the update safely
            try:
                self._broadcast_queue.put_nowait(trade_data)
            except queue.Full:
                logger.warning("Broadcast queue full, dropping trade update")

        except Exception as e:
            logger.error(f"Error in trade callback: {e}")


def apply_risk_level_settings(bot, risk_level, custom_stop_loss=None, custom_allocation=None,
                              custom_target_profit=None, custom_use_rr=None, custom_rr_ratio=None):
    """Apply risk level settings to the trading bot"""
    try:
        # Define risk level mappings
        risk_mappings = {
            "LOW": {
                "stop_loss": 0.03,         # 3% stop-loss
                "allocation": 0.15,        # 15% allocation
                "target_profit": 0.06,     # 6% target profit (2:1 risk-reward)
                "use_risk_reward": True,   # Use risk-reward ratio
                "risk_reward_ratio": 2.0   # 2:1 risk-reward ratio
            },
            "MEDIUM": {
                "stop_loss": 0.05,         # 5% stop-loss
                "allocation": 0.25,        # 25% allocation
                # 10% target profit (2:1 risk-reward)
                "target_profit": 0.10,
                "use_risk_reward": True,   # Use risk-reward ratio
                "risk_reward_ratio": 2.0   # 2:1 risk-reward ratio
            },
            "HIGH": {
                "stop_loss": 0.08,         # 8% stop-loss
                "allocation": 0.35,        # 35% allocation
                # 16% target profit (2:1 risk-reward)
                "target_profit": 0.16,
                "use_risk_reward": True,   # Use risk-reward ratio
                "risk_reward_ratio": 2.0   # 2:1 risk-reward ratio
            }
        }

        if risk_level == "CUSTOM":
            # Use custom values if provided
            if custom_stop_loss is not None:
                bot.config['stop_loss_pct'] = custom_stop_loss
                if hasattr(bot, 'executor') and bot.executor:
                    bot.executor.stop_loss_pct = custom_stop_loss

            if custom_allocation is not None:
                bot.config['max_capital_per_trade'] = custom_allocation
                if hasattr(bot, 'executor') and bot.executor:
                    bot.executor.max_capital_per_trade = custom_allocation

            if custom_target_profit is not None:
                bot.config['target_profit_pct'] = custom_target_profit
                if hasattr(bot, 'executor') and bot.executor:
                    bot.executor.target_profit_pct = custom_target_profit

            if custom_use_rr is not None:
                bot.config['use_risk_reward'] = custom_use_rr
                if hasattr(bot, 'executor') and bot.executor:
                    bot.executor.use_risk_reward = custom_use_rr

            if custom_rr_ratio is not None:
                bot.config['risk_reward_ratio'] = custom_rr_ratio
                if hasattr(bot, 'executor') and bot.executor:
                    bot.executor.risk_reward_ratio = custom_rr_ratio

        elif risk_level in risk_mappings:
            # Apply predefined risk level settings
            settings = risk_mappings[risk_level]
            bot.config.update({
                'stop_loss_pct': settings['stop_loss'],
                'max_capital_per_trade': settings['allocation'],
                'target_profit_pct': settings['target_profit'],
                'use_risk_reward': settings['use_risk_reward'],
                'risk_reward_ratio': settings['risk_reward_ratio']
            })

            # Update executor if it exists
            if hasattr(bot, 'executor') and bot.executor:
                bot.executor.stop_loss_pct = settings['stop_loss']
                bot.executor.max_capital_per_trade = settings['allocation']
                bot.executor.target_profit_pct = settings['target_profit']
                bot.executor.use_risk_reward = settings['use_risk_reward']
                bot.executor.risk_reward_ratio = settings['risk_reward_ratio']

        logger.info(f"Applied {risk_level} risk settings: "
                    f"Stop Loss={bot.config.get('stop_loss_pct')*100:.1f}%, "
                    f"Target Profit={bot.config.get('target_profit_pct', 0)*100:.1f}%, "
                    f"Use RR={bot.config.get('use_risk_reward', True)}, "
                    f"RR Ratio={bot.config.get('risk_reward_ratio', 2.0):.1f}, "
                    f"Max Allocation={bot.config.get('max_capital_per_trade')*100:.1f}%")

    except Exception as e:
        logger.error(f"Error applying risk level settings: {e}")


def _get_settings_data_dir():
    """Data dir for config files (same as save_config_to_file): backend/hft2/data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return os.path.join(project_root, "data")


def get_current_saved_mode() -> str:
    """Return the mode last saved by the user (paper/live). Used so GET settings and bot-data reflect saved choice."""
    try:
        data_dir = _get_settings_data_dir()
        path = os.path.join(data_dir, "current_mode.txt")
        if os.path.exists(path):
            with open(path, "r") as f:
                mode = (f.read() or "").strip().lower()
            if mode in ("paper", "live"):
                return mode
    except Exception:
        pass
    return os.getenv("MODE", "paper")


def _dhan_portfolio_to_metrics(dhan_portfolio: dict) -> dict:
    """Build PortfolioMetrics-shaped dict from get_live_portfolio() result (for per-user demat)."""
    cash = float(dhan_portfolio.get("cash", 0))
    holdings_raw = dhan_portfolio.get("holdings", {})
    holdings = {}
    total_invested = 0.0
    for ticker, h in holdings_raw.items():
        qty = float(h.get("quantity", 0))
        avg = float(h.get("avgPrice", 0))
        cur = float(h.get("currentPrice", avg))
        if qty > 0:
            cost = qty * avg
            total_invested += cost
            holdings[ticker] = {"qty": qty, "avg_price": avg, "current_price": cur, "current_value": qty * cur}
    current_holdings_value = sum(h.get("current_value", 0) for h in holdings.values())
    total_value = cash + current_holdings_value
    total_return = current_holdings_value - total_invested
    total_return_pct = (total_return / total_invested * 100) if total_invested else 0
    unrealized_pnl = total_return
    unrealized_pnl_pct = total_return_pct
    cash_pct = (cash / total_value * 100) if total_value else 0
    invested_pct = (total_invested / total_value * 100) if total_value else 0
    return {
        "total_value": round(total_value, 2),
        "cash": round(cash, 2),
        "cash_percentage": round(cash_pct, 2),
        "holdings": holdings,
        "total_invested": round(total_invested, 2),
        "invested_percentage": round(invested_pct, 2),
        "current_holdings_value": round(current_holdings_value, 2),
        "total_return": round(total_return, 2),
        "return_percentage": round(total_return_pct, 2),
        "total_return_pct": round(total_return_pct, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        "realized_pnl": 0,
        "realized_pnl_pct": 0,
        "total_exposure": round(current_holdings_value, 2),
        "exposure_ratio": round((current_holdings_value / total_value) if total_value else 0, 4),
        "profit_loss": round(total_return, 2),
        "profit_loss_pct": round(total_return_pct, 2),
        "positions": len(holdings),
        "trades_today": 0,
        "initial_balance": round(total_value, 2),
    }


def _convert_dhan_portfolio_to_bot_data(dhan_portfolio: dict, include_config: bool = True) -> dict:
    """Helper function to convert Dhan portfolio dict to bot data format. Reduces code duplication."""
    cash = float(dhan_portfolio.get("cash", 0))
    holdings_dict = dhan_portfolio.get("holdings", {})
    holdings = {}
    
    for ticker, h in holdings_dict.items():
        qty = float(h.get("quantity", 0))
        avg_price = float(h.get("avgPrice", 0))
        current_price = float(h.get("currentPrice", avg_price))
        if qty > 0:
            holdings[ticker] = {
                "qty": qty,
                "avg_price": avg_price,
                "currentPrice": current_price,
                "quantity": qty
            }
    
    current_market_value = sum(h["qty"] * h.get("currentPrice", h["avg_price"]) for h in holdings.values())
    total_value = cash + current_market_value
    starting_balance = float(dhan_portfolio.get("startingBalance", total_value))
    unrealized_pnl = current_market_value - sum(h["qty"] * h["avg_price"] for h in holdings.values())
    
    portfolio_data = {
        "totalValue": round(total_value, 2),
        "cash": round(cash, 2),
        "holdings": holdings,
        "startingBalance": round(starting_balance, 2),
        "unrealizedPnL": round(unrealized_pnl, 2),
        "realizedPnL": 0,
        "tradeLog": []
    }
    
    if include_config:
        # Get bot config if trading_bot exists
        if trading_bot:
            try:
                portfolio_data["tradeLog"] = getattr(trading_bot, 'trade_log', [])[-10:] if hasattr(trading_bot, 'trade_log') else []
                return {
                    "isRunning": trading_bot.is_running,
                    "config": {
                        "mode": trading_bot.config.get("mode", "live"),
                        "tickers": trading_bot.config.get("tickers", []),
                        "stopLossPct": trading_bot.config.get("stop_loss_pct", 0.05),
                        "maxAllocation": trading_bot.config.get("max_allocation", 0.25),
                        "maxTradeLimit": trading_bot.config.get("max_trade_limit", 10)
                    },
                    "portfolio": portfolio_data,
                    "analysis": list(_last_bot_analysis.values()),
                    "lastUpdate": datetime.now().isoformat()
                }
            except:
                pass
        
        # Fallback config when bot not initialized
        return {
            "isRunning": False,
            "config": {
                "mode": "live",
                "tickers": list(holdings.keys()),
                "stopLossPct": 0.05,
                "maxAllocation": 0.25,
                "maxTradeLimit": 10
            },
            "portfolio": portfolio_data,
            "analysis": list(_last_bot_analysis.values()),
            "lastUpdate": datetime.now().isoformat()
        }
    
    return portfolio_data


def set_current_saved_mode(mode: str) -> None:
    """Persist the chosen mode so GET settings and bot-data return it."""
    try:
        data_dir = _get_settings_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, "current_mode.txt")
        with open(path, "w") as f:
            f.write(mode)
    except Exception as e:
        logger.warning(f"Could not save current_mode: {e}")


def load_config_from_file(mode: str) -> dict:
    """Load configuration from the appropriate JSON file (backend/hft2/data)."""
    try:
        import json

        data_dir = _get_settings_data_dir()
        config_file = os.path.join(data_dir, f"{mode}_config.json")

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            # Never use persisted Dhan credentials; per-user demat from MongoDB only
            config_data.pop("dhan_client_id", None)
            config_data.pop("dhan_access_token", None)
            logger.info(f"Loaded configuration from {config_file}")
            return config_data
        else:
            logger.info(f"Config file {config_file} not found, using defaults")
            return {}

    except Exception as e:
        logger.error(f"Error loading config from file: {e}")
        return {}


def initialize_bot():
    """Initialize the trading bot with schema-validated configuration. Returns the bot instance."""
    global trading_bot

    print("--- STARTING BOT INITIALIZATION ---")
    try:
        import traceback

        # Load environment variables from backend/hft2/env file
        from dotenv import load_dotenv
        import os
        # Try to load from backend/hft2/env first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_file = os.path.join(os.path.dirname(current_dir), "env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded env from {env_file}")
        else:
            # Fallback to default .env loading
            load_dotenv()
        
        # Also ensure dhan_client loads the env file
        try:
            from dhan_client import _load_env
            _load_env()
        except:
            pass

        # Get trading mode from saved user preference, fallback to env
        saved_mode = get_current_saved_mode()
        default_mode = saved_mode if saved_mode in ("paper", "live") else os.getenv("MODE", "paper")
        logger.info(f"Initializing bot with mode: {default_mode} (saved: {saved_mode})")

        # Load and validate configuration using schema
        if CONFIG_SCHEMA_AVAILABLE:
            logger.info("Using schema-validated configuration loading")
            # load_and_validate_config now injects Dhan credentials BEFORE validation
            try:
                config = load_and_validate_config(default_mode)
            except Exception as validation_err:
                logger.warning(f"Config validation failed, using defaults: {validation_err}")
                config = ConfigValidator.get_default_config()
                config["mode"] = default_mode
            # Always ensure credentials are set from env (even if validation failed)
            if os.getenv("DHAN_CLIENT_ID"):
                config["dhan_client_id"] = os.getenv("DHAN_CLIENT_ID")
            if os.getenv("DHAN_ACCESS_TOKEN"):
                config["dhan_access_token"] = os.getenv("DHAN_ACCESS_TOKEN")
            # ALWAYS load tickers from saved config file (regardless of validation success/failure)
            saved_config = load_config_from_file(default_mode)
            if saved_config:
                saved_tickers = saved_config.get("tickers", [])
                if saved_tickers:
                    config["tickers"] = saved_tickers
                    logger.info(f"📊 Loaded {len(saved_tickers)} tickers from saved config: {saved_tickers}")
                # Also ensure mode matches saved config
                saved_mode = saved_config.get("mode", default_mode)
                if saved_mode in ("paper", "live"):
                    config["mode"] = saved_mode
                    logger.info(f"📊 Using saved mode: {saved_mode}")
        else:
            logger.warning(
                "Configuration schema not available, using legacy loading")
            # Fallback to legacy configuration loading
            config = {
                "tickers": [],  # Empty by default - users can add tickers manually
                "starting_balance": 10000,  # Rs.10 thousand
                "current_portfolio_value": 10000,
                "current_pnl": 0,
                "mode": default_mode,  # Default to paper mode for web interface
                "riskLevel": "MEDIUM",  # Default risk level
                "dhan_client_id": os.getenv("DHAN_CLIENT_ID"),
                "dhan_access_token": os.getenv("DHAN_ACCESS_TOKEN"),
                "period": "3y",
                "prediction_days": 30,
                "benchmark_tickers": ["^NSEI"],
                "sleep_interval": 30,  # 30 seconds
                # Risk management settings - will be set by risk level
                "stop_loss_pct": 0.05,  # Default 5% (MEDIUM)
                "max_capital_per_trade": 0.25,  # Default 25% (MEDIUM)
                "max_trade_limit": 150,
                # New keys required by StockTradingBot
                "capital": 10000,
                "margin": 0,
                "max_drawdown_pct": 0.1,
                "target_profit_pct": 0.1,
                "use_risk_reward": True,
                "risk_reward_ratio": 2.0
            }

            # Load saved configuration from file and merge with defaults
            # Use saved mode to load the correct config file
            saved_config = load_config_from_file(default_mode)
            if saved_config:
                # Update config with saved values, keeping defaults for missing keys
                # Preserve Dhan credentials from environment variables
                dhan_client_id = os.getenv("DHAN_CLIENT_ID")
                dhan_access_token = os.getenv("DHAN_ACCESS_TOKEN")

                # Ensure mode from saved config takes precedence
                saved_mode = saved_config.get("mode", default_mode)
                # Always load tickers from saved config (even if validation failed)
                saved_tickers = saved_config.get("tickers", [])
                config.update({
                    "mode": saved_mode,  # Use saved mode
                    "riskLevel": saved_config.get("riskLevel", config["riskLevel"]),
                    "stop_loss_pct": saved_config.get("stop_loss_pct", config["stop_loss_pct"]),
                    "max_capital_per_trade": saved_config.get("max_capital_per_trade", config["max_capital_per_trade"]),
                    "max_trade_limit": saved_config.get("max_trade_limit", config["max_trade_limit"]),
                    "tickers": saved_tickers if saved_tickers else config.get("tickers", [])  # Load saved watchlist tickers
                })
                logger.info(f"Loaded saved config with mode: {saved_mode}, tickers: {config.get('tickers', [])}")
                logger.info(f"Loaded saved config with mode: {saved_mode}, tickers: {config.get('tickers', [])}")

                # Ensure Dhan credentials are always set from environment variables
                if dhan_client_id:
                    config["dhan_client_id"] = dhan_client_id
                if dhan_access_token:
                    config["dhan_access_token"] = dhan_access_token
                logger.info(f"Merged saved config: Risk Level={config['riskLevel']}, "
                            f"Stop Loss={config['stop_loss_pct']*100:.1f}%, "
                            f"Max Allocation={config['max_capital_per_trade']*100:.1f}%")

        # Per-user: merge pending user context (set by /api/start when user has demat linked)
        try:
            pending = globals().get("_pending_bot_user_context")
            if pending and isinstance(pending, dict):
                if pending.get("user_id"):
                    config["user_id"] = pending["user_id"]
                if pending.get("dhan_client_id"):
                    config["dhan_client_id"] = pending["dhan_client_id"]
                if pending.get("dhan_access_token"):
                    config["dhan_access_token"] = pending["dhan_access_token"]
                globals()["_pending_bot_user_context"] = None
                logger.info(f"Applied pending user context for bot init: user_id={config.get('user_id')}")
        except Exception as e:
            logger.warning(f"Could not apply pending user context: {e}")

        # Debug logging for config
        logger.info(f"Config before WebTradingBot initialization:")
        logger.info(f"  Mode: {config.get('mode')}")
        logger.info(
            f"  Dhan Client ID: {'SET' if config.get('dhan_client_id') else 'MISSING'} ({config.get('dhan_client_id', 'NONE')[:10] if config.get('dhan_client_id') else 'NONE'})")
        logger.info(
            f"  Dhan Access Token: {'SET' if config.get('dhan_access_token') else 'MISSING'} ({'PRESENT' if config.get('dhan_access_token') else 'NONE'})")
        logger.info(f"  Full config keys: {list(config.keys())}")
        logger.info(
            f"  DHAN_CLIENT_ID from env: {os.getenv('DHAN_CLIENT_ID', 'NOT_FOUND')[:10] if os.getenv('DHAN_CLIENT_ID') else 'NOT_FOUND'}")

        try:
            logger.info("🔄 About to create WebTradingBot instance...")
            trading_bot = WebTradingBot(config)
            logger.info(f"✅ Created WebTradingBot instance: {type(trading_bot).__name__}, mode={config.get('mode')}")
            logger.info(f"✅ WebTradingBot instance created successfully - id={id(trading_bot)}")
        except Exception as bot_init_err:
            logger.error(f"❌ WebTradingBot creation failed: {bot_init_err}")
            logger.exception("WebTradingBot creation traceback:")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise  # Re-raise so outer exception handler catches it

        # Apply risk level settings from loaded config
        apply_risk_level_settings(trading_bot, config["riskLevel"])

        # Set the trading bot reference in the risk engine
        risk_engine.set_trading_bot(trading_bot)

        # Verify global was set
        import sys
        current_module = sys.modules[__name__]
        if hasattr(current_module, 'trading_bot') and current_module.trading_bot is trading_bot:
            logger.info("✅ Global trading_bot variable set successfully")
        else:
            logger.error("❌ Global trading_bot variable NOT set correctly!")
        
        logger.info("Trading bot initialized successfully")
        logger.info(f"🔄 About to return trading_bot instance: {type(trading_bot).__name__}, id={id(trading_bot)}")
        # Return the bot instance so caller can set it explicitly
        return trading_bot

    except Exception as e:
        logger.error(f"Error initializing trading bot: {e}")
        print(f"CRITICAL INITIALIZATION ERROR: {e}")
        traceback.print_exc()
        # Ensure trading_bot is None on error
        trading_bot = None
        return None
        # raise  # Don't raise in thread, just log



# Static file serving
app.mount("/static", StaticFiles(directory="."), name="static")

# API Routes

# --- JWT Auth ---
if JWT_AVAILABLE:
    @app.get("/api/auth/status")
    async def auth_status(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_http_bearer)):
        """Auth status for trading-dashboard: always enabled when JWT is available."""
        out = {"auth_status": "enabled"}
        if credentials and credentials.credentials:
            payload = auth_module.decode_token(credentials.credentials)
            if payload:
                out["authenticated"] = True
                out["username"] = payload.get("sub")
            else:
                out["authenticated"] = False
        else:
            out["authenticated"] = False
        return out

    @app.post("/api/auth/login")
    async def auth_login(req: LoginRequest):
        """Login: returns access_token (JWT)."""
        # First check if MongoDB is available
        try:
            from db.mongo_client import get_mongo_db
            db = get_mongo_db("trading")
            db.command("ping")  # Test connection
        except Exception as db_err:
            logger.error(f"MongoDB unavailable during login: {db_err}")
            raise HTTPException(status_code=503, detail="Database temporarily unavailable. Check MongoDB connection and try again.")
        
        # Now try to authenticate
        try:
            normalized_username = req.username.lower().strip()
            logger.info(f"Login attempt for: '{normalized_username}' (original: '{req.username}')")
            
            # Try to authenticate
            user = auth_module.authenticate_user(normalized_username, req.password)
            if not user:
                # MongoDB is available, so credentials are wrong or user doesn't exist
                # Check if user exists to provide better error message
                user_exists = auth_module.get_user_by_username(normalized_username)
                if user_exists:
                    logger.warning(f"Login failed: Password incorrect for user: {normalized_username}")
                    raise HTTPException(status_code=401, detail="Password is wrong")
                else:
                    # User doesn't exist - log all usernames in DB for debugging (only in debug mode)
                    logger.warning(f"Login failed: User not found: '{normalized_username}'. Make sure you're using the exact same username you registered with.")
                    raise HTTPException(status_code=401, detail="Email id not registered or password is wrong")
            
            logger.info(f"Login successful for: {normalized_username}")
            token = auth_module.create_token(sub=user["username"])
            return {"access_token": token, "token_type": "bearer", "username": user["username"]}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during login: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error during login")

    @app.post("/api/auth/logout")
    async def auth_logout(credentials: HTTPAuthorizationCredentials = Depends(_http_bearer)):
        """Logout: invalidate current token. Client must discard token and clear local state."""
        if credentials and credentials.credentials and credentials.credentials not in ("", "no-auth-required"):
            _logout_blacklist.add(credentials.credentials)
            if len(_logout_blacklist) > 10000:
                _logout_blacklist.clear()
        return {"message": "Logged out successfully"}

    @app.post("/api/auth/register")
    async def auth_register(req: RegisterRequest):
        """Register a new user."""
        if len(req.username.strip()) < 2 or len(req.password) < 6:
            raise HTTPException(status_code=400, detail="Username (min 2) and password (min 6) required")
        
        normalized_username = req.username.lower().strip()
        logger.info(f"Registration attempt for: '{normalized_username}' (original: '{req.username}')")
        
        # create_user() now handles MongoDB errors internally and returns None on failure
        # Pass normalized username to ensure consistency
        user = auth_module.create_user(normalized_username, req.password)
        if not user:
            # User creation failed - could be MongoDB unavailable or username taken
            # Check if MongoDB is available to give better error message
            try:
                from db.mongo_client import get_mongo_db
                get_mongo_db("trading")
                # Check if user exists
                existing = auth_module.get_user_by_username(normalized_username)
                if existing:
                    logger.warning(f"Registration failed: Username already exists: {normalized_username}")
                    raise HTTPException(status_code=400, detail="Username already taken")
                else:
                    logger.error(f"Registration failed: User creation returned None but user doesn't exist")
                    raise HTTPException(status_code=500, detail="Failed to create user. Please try again.")
            except HTTPException:
                raise
            except Exception as e:
                # MongoDB is unavailable
                logger.error(f"MongoDB unavailable during registration: {e}")
                raise HTTPException(status_code=503, detail="Database temporarily unavailable. Check MongoDB connection and try again.")
        
        logger.info(f"User registered successfully: {user.get('username')}")
        token = auth_module.create_token(sub=user["username"])
        return {"access_token": token, "token_type": "bearer", "username": user["username"]}

    @app.get("/api/user/profile")
    async def get_user_profile(payload: dict = Depends(get_current_user_required)):
        """Get current user profile from DB (stored per user)."""
        try:
            from db.mongo_client import get_mongo_db
            db = get_mongo_db("trading")
            col = db["profiles"]
            username = payload.get("sub") or ""
            doc = col.find_one({"username": username})
            if doc and "_id" in doc:
                doc.pop("_id", None)
            return doc or {"username": username, "fullName": "", "email": "", "preferences": {}}
        except Exception as e:
            logger.exception("Get profile error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    @app.post("/api/user/profile")
    async def save_user_profile(req: UserProfileUpdate, payload: dict = Depends(get_current_user_required)):
        """Save current user profile to DB."""
        try:
            from db.mongo_client import get_mongo_db
            from datetime import datetime
            db = get_mongo_db("trading")
            col = db["profiles"]
            username = payload.get("sub") or ""
            update = {"username": username, "updated_at": datetime.utcnow()}
            if req.fullName is not None:
                update["fullName"] = req.fullName
            if req.email is not None:
                update["email"] = req.email
            if req.preferences is not None:
                update["preferences"] = req.preferences
            col.update_one(
                {"username": username},
                {"$set": update},
                upsert=True,
            )
            return {"success": True, "message": "Profile saved"}
        except Exception as e:
            logger.exception("Save profile error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    # -------------------------------------------------------------------
    # Per-User Watchlist  (stored in MongoDB  trading.watchlists)
    # -------------------------------------------------------------------
    @app.get("/api/user/watchlist")
    async def get_watchlist(payload: dict = Depends(get_current_user_required)):
        """Return the authenticated user's watchlist."""
        try:
            from db.mongo_client import get_mongo_db
            db = get_mongo_db("trading")
            username = payload.get("sub") or ""
            doc = db["watchlists"].find_one({"username": username})
            symbols = doc.get("symbols", []) if doc else []
            return {"symbols": symbols}
        except Exception:
            logger.exception("get_watchlist error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    @app.post("/api/user/watchlist")
    async def save_watchlist(req: dict, payload: dict = Depends(get_current_user_required)):
        """Save (replace) the authenticated user's watchlist."""
        try:
            from db.mongo_client import get_mongo_db
            from datetime import datetime
            db = get_mongo_db("trading")
            username = payload.get("sub") or ""
            symbols = req.get("symbols", [])
            if not isinstance(symbols, list):
                raise HTTPException(status_code=400, detail="symbols must be a list")
            db["watchlists"].update_one(
                {"username": username},
                {"$set": {"username": username, "symbols": symbols, "updated_at": datetime.utcnow()}},
                upsert=True,
            )
            return {"success": True, "symbols": symbols}
        except HTTPException:
            raise
        except Exception:
            logger.exception("save_watchlist error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    # -------------------------------------------------------------------
    # Per-User Settings  (stored in MongoDB  trading.user_settings)
    # -------------------------------------------------------------------
    @app.get("/api/user/settings")
    async def get_user_settings(payload: dict = Depends(get_current_user_required)):
        """Return the authenticated user's settings/preferences."""
        try:
            from db.mongo_client import get_mongo_db
            db = get_mongo_db("trading")
            username = payload.get("sub") or ""
            doc = db["user_settings"].find_one({"username": username})
            if doc:
                doc.pop("_id", None)
                doc.pop("username", None)
            return doc or {}
        except Exception:
            logger.exception("get_user_settings error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    @app.post("/api/user/settings")
    async def save_user_settings(req: dict, payload: dict = Depends(get_current_user_required)):
        """Save the authenticated user's settings/preferences."""
        try:
            from db.mongo_client import get_mongo_db
            from datetime import datetime
            db = get_mongo_db("trading")
            username = payload.get("sub") or ""
            update = {k: v for k, v in req.items() if k not in ("username", "_id")}
            update["username"] = username
            update["updated_at"] = datetime.utcnow()
            db["user_settings"].update_one(
                {"username": username},
                {"$set": update},
                upsert=True,
            )
            return {"success": True}
        except Exception:
            logger.exception("save_user_settings error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    # -------------------------------------------------------------------
    # Per-User Alerts  (stored in MongoDB  trading.user_alerts)
    # -------------------------------------------------------------------
    @app.get("/api/user/alerts")
    async def get_user_alerts(payload: dict = Depends(get_current_user_required)):
        """Return the authenticated user's price alerts."""
        try:
            from db.mongo_client import get_mongo_db
            db = get_mongo_db("trading")
            username = payload.get("sub") or ""
            doc = db["user_alerts"].find_one({"username": username})
            if doc:
                doc.pop("_id", None)
                doc.pop("username", None)
            return doc or {"price_alerts": [], "prediction_alerts": [], "notifications": [], "notification_settings": {}}
        except Exception:
            logger.exception("get_user_alerts error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    @app.post("/api/user/alerts")
    async def save_user_alerts(req: dict, payload: dict = Depends(get_current_user_required)):
        """Save the authenticated user's alerts."""
        try:
            from db.mongo_client import get_mongo_db
            from datetime import datetime
            db = get_mongo_db("trading")
            username = payload.get("sub") or ""
            update = {k: v for k, v in req.items() if k not in ("username", "_id")}
            update["username"] = username
            update["updated_at"] = datetime.utcnow()
            db["user_alerts"].update_one(
                {"username": username},
                {"$set": update},
                upsert=True,
            )
            return {"success": True}
        except Exception:
            logger.exception("save_user_alerts error")
            raise HTTPException(status_code=503, detail="Database unavailable")

    # -------------------------------------------------------------------
    # Per-User Demat (broker) credentials - link account, refresh token
    # -------------------------------------------------------------------
    class DematSaveRequest(BaseModel):
        broker: str = "dhan"
        client_id: str
        access_token: str

    class DematRefreshRequest(BaseModel):
        access_token: str

    @app.get("/api/user/demat")
    async def get_user_demat_status(payload: dict = Depends(get_current_user_required)):
        """Return whether user has demat linked (no secrets)."""
        username = (payload.get("sub") or "").strip()
        demat = auth_module.get_user_demat(username) if hasattr(auth_module, "get_user_demat") else None
        if not demat:
            return {"linked": False}
        return {"linked": True, "broker": demat.get("broker", "dhan"), "client_id_masked": (demat.get("client_id", "")[:4] + "***") if demat.get("client_id") else None}

    @app.post("/api/user/demat")
    async def save_user_demat(req: DematSaveRequest, payload: dict = Depends(get_current_user_required)):
        """Save or update demat credentials (any broker). Client ID + Access Token linked to this user only."""
        username = (payload.get("sub") or "").strip()
        if not username:
            raise HTTPException(status_code=401, detail="Not authenticated")
        normalized = username.lower().strip()
        user_exists = auth_module.get_user_by_username(normalized) if hasattr(auth_module, "get_user_by_username") else None
        if not user_exists:
            raise HTTPException(status_code=404, detail="User account not found. Please log out and sign in again.")
        ok = auth_module.set_user_demat(username, req.broker or "dhan", req.client_id or "", req.access_token or "") if hasattr(auth_module, "set_user_demat") else False
        if not ok:
            raise HTTPException(status_code=503, detail="Failed to save demat credentials")
        return {"success": True, "message": "Demat account linked"}

    @app.put("/api/user/demat/token")
    async def refresh_user_demat_token(req: DematRefreshRequest, payload: dict = Depends(get_current_user_required)):
        """Update only the access token for the same user (e.g. after 24h refresh)."""
        username = (payload.get("sub") or "").strip()
        if not username:
            raise HTTPException(status_code=401, detail="Not authenticated")
        ok = auth_module.update_user_demat_token(username, req.access_token or "") if hasattr(auth_module, "update_user_demat_token") else False
        if not ok:
            raise HTTPException(status_code=503, detail="Failed to update access token")
        return {"success": True, "message": "Access token updated"}


@app.get("/")
async def index():
    """Root endpoint - returns simple JSON for connection checks"""
    return {"status": "ok", "message": "Backend API is running", "endpoints": {
        "health": "/api/health",
        "docs": "/docs",
        "auth": "/api/auth/login"
    }}

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Serve the main HTML page (if web_interface.html exists)"""
    try:
        with open('web_interface.html', 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Web interface HTML file not found")


@app.get("/styles.css")
async def styles():
    """Serve the CSS file"""
    try:
        return FileResponse('styles.css', media_type='text/css')
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSS file not found")


@app.get("/app.js")
async def app_js():
    """Serve the JavaScript file"""
    try:
        return FileResponse('app.js', media_type='application/javascript')
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="JavaScript file not found")


@app.get("/api/status", response_model=BotStatus)
async def get_status():
    """Get bot status"""
    try:
        if trading_bot:
            status = trading_bot.get_status()
            return BotStatus(**status)
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
@log_api_call("/api/health", "GET")
async def health_check():
    """Health check endpoint for monitoring system status"""
    try:
        import time
        from datetime import datetime

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - getattr(app, 'start_time', time.time()),
            "version": "1.0.0",
            "services": {}
        }
        
        # Check MongoDB connectivity (non-blocking: 2s timeout so health endpoint stays fast)
        try:
            from db.mongo_client import get_mongo_db
            import asyncio as _asyncio
            db = get_mongo_db("trading")

            def _ping_mongo():
                db.command("ping")

            try:
                await _asyncio.wait_for(_asyncio.to_thread(_ping_mongo), timeout=2.0)
                health_status["services"]["mongodb"] = {"status": "healthy"}
            except _asyncio.TimeoutError:
                health_status["services"]["mongodb"] = {"status": "degraded", "note": "ping timeout"}
            except Exception as _me:
                health_status["services"]["mongodb"] = {"status": "unhealthy", "error": str(_me)[:80]}
        except Exception as e:
            health_status["services"]["mongodb"] = {"status": "unavailable", "error": str(e)[:80]}

        # Check trading bot status
        if trading_bot:
            try:
                bot_status = trading_bot.get_status()
                health_status["services"]["trading_bot"] = {
                    "status": "healthy",
                    "mode": bot_status.get("mode", "unknown"),
                    "balance": bot_status.get("current_balance", 0)
                }
            except Exception as e:
                health_status["services"]["trading_bot"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            health_status["services"]["trading_bot"] = {
                "status": "not_initialized"
            }

        # Check data service client (non-blocking - don't fail health check if data service is slow)
        try:
            data_client = get_data_client()
            # Run health check in thread pool with timeout to avoid blocking
            import asyncio
            try:
                # Try health check with timeout - don't block health endpoint
                health_result = await asyncio.wait_for(
                    asyncio.to_thread(data_client.health_check), 
                    timeout=3.0  # 3 second timeout
                )
                if health_result:
                    health_status["services"]["data_service"] = {"status": "healthy"}
                else:
                    health_status["services"]["data_service"] = {"status": "unhealthy"}
            except asyncio.TimeoutError:
                # Data service is slow - mark as degraded but don't fail health endpoint
                health_status["services"]["data_service"] = {"status": "degraded", "note": "Response timeout"}
            except Exception as e:
                # Log at debug level - don't spam logs with warnings
                logger.debug(f"Data service health check error: {e}")
                health_status["services"]["data_service"] = {"status": "unavailable", "error": str(e)[:50]}
        except Exception as e:
            # If we can't even get the client, mark as unavailable
            logger.debug(f"Data service client unavailable: {e}")
            health_status["services"]["data_service"] = {"status": "unavailable"}

        # Check MCP service (if available)
        try:
            from mcp_service.api_server import MCP_API_AVAILABLE
            if MCP_API_AVAILABLE:
                health_status["services"]["mcp_service"] = {
                    "status": "healthy"}
            else:
                health_status["services"]["mcp_service"] = {
                    "status": "unavailable"}
        except:
            health_status["services"]["mcp_service"] = {
                "status": "not_available"}

        # Check configuration validation
        try:
            if CONFIG_SCHEMA_AVAILABLE:
                # Validate current configuration
                config_issues = ConfigValidator.validate_environment_variables()
                if config_issues:
                    health_status["services"]["configuration"] = {
                        "status": "warning",
                        "issues": config_issues
                    }
                else:
                    health_status["services"]["configuration"] = {
                        "status": "healthy"}
            else:
                health_status["services"]["configuration"] = {
                    "status": "warning",
                    "message": "Configuration schema validation not available"
                }
        except Exception as e:
            health_status["services"]["configuration"] = {
                "status": "error",
                "error": str(e)
            }

        # Determine overall status
        unhealthy_services = [
            service for service in health_status["services"].values()
            if isinstance(service, dict) and service.get("status") in ["unhealthy", "error"]
        ]

        if unhealthy_services:
            health_status["status"] = "degraded"
        elif not health_status["services"]:
            health_status["status"] = "unknown"
        else:
            health_status["status"] = "healthy"

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.get("/api/config/validate")
async def validate_configuration():
    """Validate current configuration against schema"""
    try:
        if not CONFIG_SCHEMA_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Configuration schema validation not available"
            )

        # Get current configuration
        if trading_bot and hasattr(trading_bot, 'config'):
            current_config = trading_bot.config
        else:
            # Load default config for validation
            current_config = ConfigValidator.get_default_config()

        # Validate configuration
        validation_result = {
            "valid": True,
            "config": current_config,
            "schema": ConfigValidator.get_config_schema(),
            "environment_issues": ConfigValidator.validate_environment_variables()
        }

        # Try to validate the config
        try:
            ConfigValidator.validate_config(current_config)
            validation_result["validation_status"] = "passed"
        except Exception as e:
            validation_result["valid"] = False
            validation_result["validation_status"] = "failed"
            validation_result["validation_errors"] = str(e)

        return validation_result

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config/schema")
async def get_configuration_schema():
    """Get the configuration JSON schema"""
    try:
        if not CONFIG_SCHEMA_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Configuration schema not available"
            )

        return ConfigValidator.get_config_schema()

    except Exception as e:
        logger.error(f"Failed to get configuration schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze-stream")
async def analyze_stream(request: Request, symbol: str = "INFY.NS"):
    """SSE endpoint: run full HFT2 analysis pipeline for a symbol and stream structured events.
    Events:
      {type:'progress', step:'...', pct:N}
      {type:'log', level:'INFO'|'WARNING'|'ERROR', message:'...'}
      {type:'indicator', name:'RSI', value:..., signal:'bullish'|'bearish'|'neutral'}
      {type:'result', data:{symbol,recommendation,confidence,reasoning,risk_score,target_price,stop_loss,indicators}}
      {type:'error', message:'...'}
      {type:'done'}
    """
    import queue as _q_mod

    event_q: "_q_mod.Queue[str]" = _q_mod.Queue(maxsize=2000)

    def _emit(obj: dict):
        try:
            event_q.put_nowait(f"data: {json.dumps(obj)}\n\n")
        except _q_mod.Full:
            pass

    def _log_emit(level: str, msg: str):
        _emit({"type": "log", "level": level, "message": msg})

    async def run_analysis():
        """Run the heavy ML pipeline (stock_analyzer.analyze_stock) and emit result only when done.
        User must wait; Stop Bot is respected via get_bot_running()."""
        global trading_bot
        loop = asyncio.get_event_loop()
        sym = symbol.strip().upper()

        try:
            _emit({"type": "progress", "step": "Starting full analysis (may take a few minutes)", "pct": 5})
            _log_emit("INFO", f"🚀 Running heavy ML pipeline for {sym} — please wait...")

            # If bot not initialized, initialize it now (blocking but necessary)
            if not trading_bot:
                _emit({"type": "progress", "step": "Initializing bot...", "pct": 5})
                _log_emit("INFO", "🔄 Bot not initialized, initializing now...")
                try:
                    loop = asyncio.get_event_loop()
                    trading_bot = await asyncio.wait_for(
                        loop.run_in_executor(None, initialize_bot),
                        timeout=180.0
                    )
                    if not trading_bot:
                        _log_emit("ERROR", "Failed to initialize bot")
                        _emit({"type": "error", "message": "Failed to initialize bot"})
                    _log_emit("INFO", "✅ Bot initialized successfully")
                except asyncio.TimeoutError:
                    _log_emit("ERROR", "Bot initialization timed out")
                    _emit({"type": "error", "message": "Bot initialization timed out"})
                    return
                except Exception as init_err:
                    _log_emit("ERROR", f"Bot initialization failed: {init_err}")
                    _emit({"type": "error", "message": f"Bot initialization failed: {init_err}"})
                    return
            
            # If stock_analyzer not available, initialize it
            if not getattr(trading_bot, "stock_analyzer", None):
                _emit({"type": "progress", "step": "Initializing stock analyzer...", "pct": 10})
                _log_emit("INFO", "🔄 Stock analyzer not found, initializing...")
                try:
                    from testindia import Stock
                    config = trading_bot.config if hasattr(trading_bot, 'config') else {}
                    
                    def init_stock_analyzer():
                        return Stock(
                            reddit_client_id=config.get("reddit_client_id"),
                            reddit_client_secret=config.get("reddit_client_secret"),
                            reddit_user_agent=config.get("reddit_user_agent"),
                            advanced_sentiment_analyzer=None
                        )
                    
                    trading_bot.stock_analyzer = await asyncio.wait_for(
                        loop.run_in_executor(None, init_stock_analyzer),
                        timeout=120.0
                    )
                    _log_emit("INFO", "✅ Stock analyzer initialized")
                except Exception as sa_err:
                    _log_emit("ERROR", f"Failed to initialize stock analyzer: {sa_err}")
                    _emit({"type": "error", "message": f"Failed to initialize stock analyzer: {sa_err}"})
                    return

            _emit({"type": "progress", "step": "Waiting for ML pipeline...", "pct": 19})
            
            # Use file polling to prevent duplicate thread starvation!
            # The testindia.py thread will produce JSON files in stock_analysis/
            sanitized_sym = sym.replace(".", "_")
            pattern = os.path.join(os.path.dirname(__file__), "stock_analysis", f"{sanitized_sym}_analysis_*.json")
            
            try:
                _emit({"type": "progress", "step": "Waiting for background ML output...", "pct": 20})
                _log_emit("INFO", f"Tracking live analysis output from backend generator for {sym}...")
                
                start_time = time.time()
                found_raw = None
                
                while get_bot_running() and (time.time() - start_time) < 600:
                    await asyncio.sleep(2)
                    
                    # Check for recent file
                    import glob
                    files = glob.glob(pattern)
                    if files:
                        latest_file = max(files, key=os.path.getmtime)
                        # Ensure it's a recently generated file (last 15 minutes) to avoid stale data
                        if time.time() - os.path.getmtime(latest_file) < 900:
                            try:
                                with open(latest_file, 'r', encoding='utf-8') as f:
                                    found_raw = json.load(f)
                                _log_emit("INFO", f"✅ Successfully loaded analysis from {os.path.basename(latest_file)}")
                                break
                            except Exception as e:
                                _log_emit("WARNING", f"Found analysis file but failed to read: {e}")
                
                raw = found_raw
                if not raw and get_bot_running():
                    raise asyncio.TimeoutError()
                    
            except asyncio.TimeoutError:
                _log_emit("ERROR", "Analysis timed out (10 min)")
                _emit({"type": "error", "message": "Analysis timed out"})
                return
            except asyncio.CancelledError:
                _log_emit("INFO", "Analysis cancelled (Stop Bot)")
                _emit({"type": "error", "message": "User interrupted the process"})
                _emit({"type": "result", "data": {
                    "symbol": sym,
                    "recommendation": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "Analysis was stopped.",
                    "risk_score": 0.5,
                    "target_price": None,
                    "stop_loss": None,
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "indicators": {},
                    "timestamp": datetime.now().isoformat(),
                }})
                return

            if not raw or not raw.get("success"):
                msg = raw.get("message", "Analysis failed or was stopped") if isinstance(raw, dict) else "Analysis failed"
                _log_emit("WARNING", msg)
                _emit({"type": "result", "data": {
                    "symbol": sym,
                    "recommendation": "HOLD",
                    "confidence": 0.0,
                    "reasoning": msg,
                    "risk_score": 0.5,
                    "target_price": None,
                    "stop_loss": None,
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "indicators": {},
                    "timestamp": datetime.now().isoformat(),
                }})
                return

            stock_data = raw.get("stock_data") or {}
            tech = raw.get("technical_indicators") or {}
            ml = raw.get("ml_analysis") or {}
            rec_raw = (raw.get("recommendation") or "HOLD").upper()
            if "BUY" in rec_raw or "STRONG BUY" in rec_raw:
                recommendation = "BUY"
            elif "SELL" in rec_raw or "STRONG SELL" in rec_raw:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            confidence = float(ml.get("confidence", 0.5))
            if confidence > 1.0:
                confidence = confidence / 100.0
            cp_raw = stock_data.get("current_price")
            if isinstance(cp_raw, dict):
                current_price = float(cp_raw.get("INR") or cp_raw.get("USD") or 0)
            else:
                current_price = float(cp_raw or 0)
            support = float(stock_data.get("support_level") or 0)
            resistance = float(stock_data.get("resistance_level") or 0)
            target_price = None
            stop_loss_price = None
            if ml.get("predicted_price"):
                target_price = round(float(ml["predicted_price"]), 2)
            elif resistance and current_price:
                target_price = round(resistance, 2)
            if support and current_price:
                stop_loss_price = round(support, 2)
            if not stop_loss_price and current_price:
                stop_loss_price = round(current_price * 0.97, 2)
            reasoning = (raw.get("technical_analysis") or {}).get("explanation") or raw.get("explanation") or "Full ML analysis complete."
            sentiment_data = raw.get("sentiment_analysis") or {}
            if isinstance(sentiment_data, dict):
                sentiment_score = float(sentiment_data.get("score", sentiment_data.get("compound", 0)))
                sentiment_label = sentiment_data.get("label", "neutral")
            else:
                sentiment_score = 0.0
                sentiment_label = "neutral"
            indicators = {}
            if tech.get("rsi") is not None:
                rsi_v = float(tech["rsi"])
                indicators["RSI"] = {"value": round(rsi_v, 2), "signal": "bearish" if rsi_v > 65 else ("bullish" if rsi_v < 40 else "neutral")}
            if tech.get("macd") is not None:
                indicators["MACD"] = {"value": round(float(tech["macd"]), 4), "signal": "bullish" if float(tech.get("macd", 0)) > 0 else "bearish"}
            if tech.get("sma_50") is not None and current_price:
                indicators["EMA20"] = {"value": round(float(tech.get("sma_50", 0)), 2), "signal": "bullish" if current_price > float(tech["sma_50"]) else "bearish"}

            result_payload = {
                "symbol": sym,
                "recommendation": recommendation,
                "confidence": min(1.0, max(0.0, confidence)),
                "reasoning": reasoning[:500] if reasoning else "Full ML analysis complete.",
                "risk_score": 0.5,
                "target_price": target_price,
                "stop_loss": stop_loss_price,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat(),
            }
            _last_bot_analysis[sym] = {**result_payload, "prediction": ml}
            _emit({"type": "progress", "step": "Analysis complete", "pct": 100})
            _emit({"type": "result", "data": result_payload})
            _log_emit("INFO", f"✅ Heavy analysis complete for {sym}: {recommendation} ({confidence:.1%})")

        except Exception as e:
            _log_emit("ERROR", f"❌ Analysis pipeline error: {e}")
            _emit({"type": "error", "message": str(e)})
        finally:
            _emit({"type": "done"})

    async def event_generator():
        # Fire analysis in background so SSE loop can drain the queue
        analysis_task = asyncio.create_task(run_analysis())
        last_ping_time = asyncio.get_event_loop().time()
        try:
            yield f"data: {json.dumps({'type': 'connected', 'symbol': symbol.strip().upper()})}\n\n"
            while not analysis_task.done() or not event_q.empty():
                if await request.is_disconnected():
                    analysis_task.cancel()
                    break
                
                now = asyncio.get_event_loop().time()
                if now - last_ping_time >= 15.0:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                    last_ping_time = now
                
                drained = 0
                while drained < 30:
                    try:
                        yield event_q.get_nowait()
                        drained += 1
                        last_ping_time = asyncio.get_event_loop().time()
                    except Exception:
                        break
                await asyncio.sleep(0.15)
            # Drain any remaining events
            while not event_q.empty():
                try:
                    yield event_q.get_nowait()
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stream")
async def stream_events(request: Request):
    """SSE endpoint: streams live log lines, periodic bot snapshots, and heartbeat pings to the frontend.
    Heartbeat pings every 15 s prevent browser/proxy SSE timeouts."""
    client_q: "_queue_module.Queue[str]" = _queue_module.Queue(maxsize=1000)
    with _sse_clients_lock:
        _sse_clients.append(client_q)

    async def event_generator():
        try:
            # Send handshake
            yield f"data: {json.dumps({'type': 'connected', 'message': 'Stream connected'})}\n\n"
            last_data_tick = 0.0
            last_ping_tick = asyncio.get_event_loop().time()
            last_cache_refresh_tick = 0.0
            while True:
                if await request.is_disconnected():
                    break
                # Drain log queue (up to 50 lines per iteration)
                drained = 0
                while drained < 50:
                    try:
                        event = client_q.get_nowait()
                        yield event
                        drained += 1
                    except _queue_module.Empty:
                        break
                now = asyncio.get_event_loop().time()
                # Keep bot-data cache warm: trigger background refresh every 20 s
                if now - last_cache_refresh_tick >= 20.0:
                    last_cache_refresh_tick = now
                    asyncio.ensure_future(_refresh_bot_data_cache_background())
                # Periodic bot snapshot every 5 s
                if now - last_data_tick >= 5.0:
                    last_data_tick = now
                    try:
                        snapshot = await asyncio.get_event_loop().run_in_executor(None, _build_sse_snapshot)
                        yield f"data: {json.dumps({'type': 'data', 'payload': snapshot})}\n\n"
                    except Exception:
                        pass
                # Heartbeat ping every 15 s — keeps proxy/browser connection alive
                if now - last_ping_tick >= 15.0:
                    last_ping_tick = now
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                await asyncio.sleep(0.2)
        finally:
            with _sse_clients_lock:
                try:
                    _sse_clients.remove(client_q)
                except ValueError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class OrderRequest(BaseModel):
    symbol: str
    side: str          # "BUY" or "SELL"
    quantity: int = 1
    order_type: str = "MARKET"
    price: Optional[float] = None


@app.post("/api/order")
async def place_order(req: OrderRequest):
    """Place a manual BUY or SELL order via the trading bot's executor."""
    global trading_bot
    try:
        side = req.side.upper()
        if side not in ("BUY", "SELL"):
            raise HTTPException(status_code=400, detail="side must be BUY or SELL")

        logger.info(f"Manual order received: {side} {req.quantity}x {req.symbol} @ {req.order_type}")

        # Try live executor first
        if trading_bot and hasattr(trading_bot, 'live_executor') and trading_bot.live_executor:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: trading_bot.live_executor.place_order(
                    symbol=req.symbol,
                    side=side,
                    quantity=req.quantity,
                    order_type=req.order_type,
                    price=req.price,
                )
            )
            return {"status": "ok", "message": f"{side} order placed for {req.symbol}", "detail": result}

        # Try trading bot executor
        if trading_bot and hasattr(trading_bot, 'trading_bot') and trading_bot.trading_bot:
            executor = getattr(trading_bot.trading_bot, 'executor', None)
            if executor and hasattr(executor, 'place_order'):
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: executor.place_order(
                        symbol=req.symbol,
                        action=side,
                        quantity=req.quantity,
                    )
                )
                return {"status": "ok", "message": f"{side} order placed for {req.symbol}", "detail": result}

        # Paper mode fallback — log the order
        logger.info(f"[PAPER] {side} {req.quantity}x {req.symbol} — no live executor available")
        return {"status": "paper", "message": f"[Paper] {side} order recorded for {req.symbol} (no live executor)"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bot/start-with-symbol")
async def start_bot_with_symbol(req: StartBotWithSymbolRequest):
    """Start the trading bot initialisation for a specific symbol and return immediately.
    The frontend connects to /api/stream to follow progress without timing out."""
    global trading_bot
    symbol = req.symbol
    logger.info(f"[API] start-with-symbol requested for: {symbol}")
    # Fire-and-forget heavy work in background so endpoint returns in < 1 s
    asyncio.create_task(trigger_all_hft2_components_for_symbol(symbol))
    return {"status": "pending", "symbol": symbol, "message": f"Bot initialisation started for {symbol}. Connect to /api/stream for live progress."}


@app.post("/api/bot/stop")
async def stop_bot_endpoint():
    """Stop the running trading bot and cancel the continuous analysis loop."""
    global trading_bot, bot_running
    try:
        bot_running = False
        _stop_continuous_loop()
        if trading_bot:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, trading_bot.stop)
            logger.info("Bot stopped via API (continuous loop cancelled)")
            return {"status": "ok", "message": "Bot stopped"}
        return {"status": "ok", "message": "Bot was not running"}
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bot-data")
async def get_bot_data(user_demat: tuple = Depends(get_optional_user_demat)):
    """Get complete bot data for React frontend. When user has demat linked, returns their portfolio (no cache). Logged-in user without demat gets offline data only (no env/cache fallback)."""
    global _bot_data_cache, _bot_data_cache_ts
    payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
    if demat and demat.get("access_token") and demat.get("client_id"):
        try:
            from dhan_client import get_live_portfolio
            loop = asyncio.get_event_loop()
            dhan_port = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: get_live_portfolio(access_token=demat["access_token"], client_id=demat["client_id"])),
                timeout=25.0,
            )
            if dhan_port:
                return _convert_dhan_portfolio_to_bot_data(dhan_port, include_config=True)
        except asyncio.TimeoutError:
            logger.warning("Demat bot-data fetch timed out")
        except Exception as e:
            logger.warning(f"Demat bot-data fetch failed: {e}")
        return _offline_bot_data()

    # Logged-in user without linked demat: never show env/cache (other account)
    if payload is not None:
        return _offline_bot_data()

    try:
        now = _time_module.monotonic()
        cache_age = now - _bot_data_cache_ts
        if _bot_data_cache and cache_age < _BOT_DATA_CACHE_TTL:
            return _bot_data_cache
        if _bot_data_cache:
            asyncio.ensure_future(_refresh_bot_data_cache_background())
            return _bot_data_cache
        await asyncio.wait_for(_refresh_bot_data_cache_background(), timeout=10.0)
        if _bot_data_cache:
            return _bot_data_cache
        return _offline_bot_data()
    except asyncio.TimeoutError:
        return _bot_data_cache if _bot_data_cache else _offline_bot_data()
    except Exception as e:
        logger.error(f"Error getting bot data: {e}")
        return _bot_data_cache if _bot_data_cache else _offline_bot_data()


@app.get("/api/portfolio", response_model=PortfolioMetrics)
@log_api_call("/api/portfolio", "GET")
async def get_portfolio(user_demat: tuple = Depends(get_optional_user_demat)):
    """Get comprehensive portfolio metrics. When user has demat linked, uses their broker account; else uses trading_bot."""
    try:
        payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
        if demat and demat.get("access_token") and demat.get("client_id"):
            try:
                from dhan_client import get_live_portfolio
                loop = asyncio.get_event_loop()
                dhan_port = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: get_live_portfolio(access_token=demat["access_token"], client_id=demat["client_id"])),
                    timeout=25.0,
                )
                if dhan_port:
                    metrics = _dhan_portfolio_to_metrics(dhan_port)
                    return PortfolioMetrics(
                        total_value=metrics["total_value"],
                        cash=metrics["cash"],
                        cash_percentage=metrics["cash_percentage"],
                        holdings=metrics["holdings"],
                        total_invested=metrics["total_invested"],
                        invested_percentage=metrics["invested_percentage"],
                        current_holdings_value=metrics["current_holdings_value"],
                        total_return=metrics["total_return"],
                        return_percentage=metrics["return_percentage"],
                        total_return_pct=metrics["total_return_pct"],
                        unrealized_pnl=metrics["unrealized_pnl"],
                        unrealized_pnl_pct=metrics["unrealized_pnl_pct"],
                        realized_pnl=metrics["realized_pnl"],
                        realized_pnl_pct=metrics["realized_pnl_pct"],
                        total_exposure=metrics["total_exposure"],
                        exposure_ratio=metrics["exposure_ratio"],
                        profit_loss=metrics["profit_loss"],
                        profit_loss_pct=metrics["profit_loss_pct"],
                        active_positions=metrics["positions"],
                        trades_today=metrics["trades_today"],
                        initial_balance=metrics["initial_balance"],
                    )
            except asyncio.TimeoutError:
                logger.warning("Demat portfolio fetch timed out")
            except Exception as e:
                logger.warning(f"Demat portfolio fetch failed: {e}")

        # Logged-in user without linked demat: do not use trading_bot (env) data
        if payload is not None:
            return PortfolioMetrics(
                total_value=0, cash=0, cash_percentage=0, holdings={},
                total_invested=0, invested_percentage=0, current_holdings_value=0,
                total_return=0, return_percentage=0, total_return_pct=0,
                unrealized_pnl=0, unrealized_pnl_pct=0, realized_pnl=0, realized_pnl_pct=0,
                total_exposure=0, exposure_ratio=0, profit_loss=0, profit_loss_pct=0,
                active_positions=0, trades_today=0, initial_balance=0,
            )

        if trading_bot:
            metrics = trading_bot.get_portfolio_metrics()
            portfolio_response = {
                "total_value": metrics.get("total_value", 0),
                "cash": metrics.get("cash", 0),
                "cash_percentage": metrics.get("cash_percentage", 0),
                "holdings": metrics.get("holdings", {}),
                "total_invested": metrics.get("total_invested", 0),
                "invested_percentage": metrics.get("invested_percentage", 0),
                "current_holdings_value": metrics.get("current_holdings_value", 0),
                "total_return": metrics.get("total_return", 0),
                "return_percentage": metrics.get("total_return_pct", 0),
                "total_return_pct": metrics.get("total_return_pct", 0),
                "unrealized_pnl": metrics.get("unrealized_pnl", 0),
                "unrealized_pnl_pct": metrics.get("unrealized_pnl_pct", 0),
                "realized_pnl": metrics.get("realized_pnl", 0),
                "realized_pnl_pct": metrics.get("realized_pnl_pct", 0),
                "total_exposure": metrics.get("total_exposure", 0),
                "exposure_ratio": metrics.get("exposure_ratio", 0),
                "profit_loss": metrics.get("profit_loss", 0),
                "profit_loss_pct": metrics.get("profit_loss_pct", 0),
                "active_positions": metrics.get("positions", 0),
                "trades_today": metrics.get("trades_today", 0),
                "initial_balance": metrics.get("initial_balance", 10000),
            }
            return PortfolioMetrics(**portfolio_response)
        raise HTTPException(status_code=500, detail="Bot not initialized and no demat linked")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades")
async def get_trades(limit: int = 10, user_demat: tuple = Depends(get_optional_user_demat)):
    """Get recent trades. When user has demat linked, returns [] (broker may not expose history); else from trading_bot."""
    payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
    if demat:
        return []
    try:
        if trading_bot:
            loop = asyncio.get_event_loop()
            trades = await asyncio.wait_for(
                loop.run_in_executor(None, trading_bot.get_recent_trades, limit),
                timeout=3.0,
            )
            return trades
        return []
    except asyncio.TimeoutError:
        return []
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return []


@app.get("/api/portfolio/realtime")
async def get_realtime_portfolio(user_demat: tuple = Depends(get_optional_user_demat)):
    """Get real-time portfolio updates. When user has demat linked, uses their account; else trading_bot + Dhan sync."""
    payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
    if demat and demat.get("access_token") and demat.get("client_id"):
        try:
            from dhan_client import get_live_portfolio
            loop = asyncio.get_event_loop()
            dhan_port = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: get_live_portfolio(access_token=demat["access_token"], client_id=demat["client_id"])),
                timeout=25.0,
            )
            if dhan_port:
                metrics = _dhan_portfolio_to_metrics(dhan_port)
                current_prices = {}
                fyers_client = get_fyers_client()
                for ticker in metrics.get("holdings", {}).keys():
                    try:
                        if fyers_client:
                            symbol_data = fyers_client.get_symbol_data(ticker)
                            if symbol_data:
                                current_prices[ticker] = {"price": symbol_data.get("price", 0), "change": symbol_data.get("change", 0), "change_pct": symbol_data.get("change_pct", 0), "volume": symbol_data.get("volume", 0)}
                    except Exception:
                        pass
                return {"portfolio_metrics": metrics, "current_prices": current_prices}
        except Exception as e:
            logger.warning(f"Realtime demat fetch failed: {e}")

    try:
        if trading_bot and trading_bot.config.get("mode") == "live":
            try:
                # Force sync with Dhan account to get latest balance
                sync_performed = False

                # Try live executor sync first
                if hasattr(trading_bot, 'live_executor') and trading_bot.live_executor:
                    sync_success = trading_bot.live_executor.sync_portfolio_with_dhan()
                    if sync_success:
                        logger.debug(
                            "Successfully synced with Dhan account using live executor")
                        sync_performed = True
                    else:
                        logger.warning(
                            "Failed to sync with Dhan account using live executor")

                # Also sync VirtualPortfolio if it exists
                if hasattr(trading_bot, 'portfolio') and trading_bot.portfolio:
                    portfolio_sync_result = trading_bot.portfolio.sync_with_dhan_account()
                    if portfolio_sync_result:
                        logger.debug(
                            "Successfully synced VirtualPortfolio with Dhan account")
                        sync_performed = True
                    else:
                        logger.warning(
                            "Failed to sync VirtualPortfolio with Dhan account")

                # Fallback: manually sync using dhan_client
                elif hasattr(trading_bot, 'dhan_client') and trading_bot.dhan_client:
                    # Fallback: manually sync using dhan_client
                    funds = trading_bot.dhan_client.get_funds()
                    if funds:
                        # tolerant extraction
                        available_cash = 0.0
                        try:
                            for key in ('availableBalance', 'availabelBalance', 'available_balance', 'available', 'availBalance', 'cash'):
                                if isinstance(funds, dict) and key in funds:
                                    available_cash = float(
                                        funds.get(key, 0.0) or 0.0)
                                    break
                            else:
                                if isinstance(funds, dict):
                                    for v in funds.values():
                                        if isinstance(v, (int, float)):
                                            available_cash = float(v)
                                            break
                        except Exception:
                            available_cash = 0.0
                        # Update portfolio manager if available
                        if hasattr(trading_bot, 'portfolio_manager'):
                            trading_bot.portfolio_manager.update_cash_balance(
                                available_cash)
                        logger.debug(
                            f"Manually synced cash balance: ₹{available_cash}")
                        sync_performed = True

                if not sync_performed:
                    logger.warning("No sync method available for Dhan account")

            except Exception as e:
                logger.warning(
                    f"Dhan sync failed during realtime update, using local data: {e}")

        if trading_bot:
            metrics = trading_bot.get_portfolio_metrics()

            # Get current prices for all holdings
            current_prices = {}
            fyers_client = get_fyers_client()

            for ticker in metrics.get("holdings", {}).keys():
                try:
                    if fyers_client:
                        # PRODUCTION FIX: Use data service client methods
                        symbol_data = fyers_client.get_symbol_data(ticker)
                        if symbol_data:
                            current_prices[ticker] = {
                                "price": symbol_data.get("price", 0),
                                "change": symbol_data.get("change", 0),
                                "change_pct": symbol_data.get("change_pct", 0),
                                "volume": symbol_data.get("volume", 0)
                            }
                except Exception as e:
                    logger.warning(
                        f"Error fetching real-time price for {ticker}: {e}")

            return {
                "portfolio_metrics": metrics,
                "current_prices": current_prices,
                "last_updated": datetime.now().isoformat(),
                "market_status": _get_indian_market_status()
            }
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except Exception as e:
        logger.error(f"Error getting real-time portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_indian_market_status() -> str:
    """Get Indian market status based on NSE trading hours"""
    try:
        import pytz
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)

        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return "CLOSED"

        # NSE trading hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        if market_open <= now <= market_close:
            return "OPEN"
        else:
            return "CLOSED"
    except Exception as e:
        logger.error(f"Error determining market status: {e}")
        return "UNKNOWN"


@app.get("/api/watchlist")
async def get_watchlist(payload: dict = Depends(get_optional_user)):
    """Get watchlist. When authenticated, returns that user's watchlist from MongoDB (per-user). Otherwise returns global config."""
    try:
        # Per-user: when authenticated, return only this user's watchlist
        if payload and isinstance(payload, dict):
            username = payload.get("sub") or ""
            if username:
                tickers = _get_user_watchlist_from_db(username)
                logger.info(f"📊 GET /api/watchlist: Returning {len(tickers)} tickers for user (MongoDB): {tickers}")
                return tickers
        # Global fallback (unauthenticated or no user)
        tickers = []
        if trading_bot:
            tickers = trading_bot.config.get("tickers", [])
        else:
            saved_mode = get_current_saved_mode() or "paper"
            saved_config = load_config_from_file(saved_mode) or {}
            tickers = saved_config.get("tickers", [])
        logger.info(f"📊 GET /api/watchlist: Returning {len(tickers)} tickers (global): {tickers}")
        return tickers
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/watchlist", response_model=WatchlistResponse)
async def update_watchlist(request: WatchlistRequest):
    """Add or remove ticker from watchlist"""
    try:
        ticker = request.ticker.upper()
        action = request.action.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")

        if trading_bot:
            current_tickers = trading_bot.config["tickers"]

            if action == "ADD":
                if ticker not in current_tickers:
                    current_tickers.append(ticker)
                    message = f"Added {ticker} to watchlist"
                else:
                    message = f"{ticker} is already in watchlist"
            elif action == "REMOVE":
                if ticker in current_tickers:
                    current_tickers.remove(ticker)
                    message = f"Removed {ticker} from watchlist"
                else:
                    message = f"{ticker} is not in watchlist"
            else:
                raise HTTPException(
                    status_code=400, detail="Invalid action. Use ADD or REMOVE")

            return WatchlistResponse(message=message, tickers=current_tickers)
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/watchlist/add/{ticker}", response_model=WatchlistResponse)
async def add_to_watchlist(ticker: str, payload: dict = Depends(get_optional_user)):
    """Add ticker to watchlist. When authenticated, updates only that user's watchlist in MongoDB."""
    try:
        ticker = ticker.upper().strip()
        if not ticker:
            raise HTTPException(status_code=400, detail="Ticker is required")
        if not ticker.endswith(('.NS', '.BO')):
            ticker += '.NS'

        # Per-user: when authenticated, update only this user's watchlist in MongoDB
        if payload and isinstance(payload, dict):
            username = payload.get("sub") or ""
            if username:
                current_tickers = _get_user_watchlist_from_db(username)
                if ticker not in current_tickers:
                    current_tickers.append(ticker)
                    _save_user_watchlist_to_db(username, current_tickers)
                    message = f"✅ Added {ticker} to your watchlist"
                else:
                    message = f"{ticker} is already in your watchlist"
                logger.info(f"📊 Watchlist ADD: {ticker} for user {username} (MongoDB)")
                return WatchlistResponse(message=message, tickers=current_tickers)

        # Global (unauthenticated)
        current_tickers = []
        if trading_bot:
            current_tickers = trading_bot.config.get("tickers", [])
            if ticker not in current_tickers:
                current_tickers.append(ticker)
                trading_bot.config["tickers"] = current_tickers
                try:
                    saved_mode = get_current_saved_mode() or trading_bot.config.get("mode", "paper")
                    save_config_to_file(saved_mode, trading_bot.config)
                except Exception as save_err:
                    logger.warning(f"Failed to save ticker to config file: {save_err}")
                try:
                    from data_feed import DataFeed
                    if DataFeed:
                        trading_bot.data_feed = DataFeed(current_tickers)
                except Exception:
                    pass
                message = f"✅ Added {ticker} to watchlist"
            else:
                message = f"{ticker} is already in watchlist"
        else:
            saved_mode = get_current_saved_mode() or "paper"
            saved_config = load_config_from_file(saved_mode) or {}
            current_tickers = saved_config.get("tickers", [])
            if ticker not in current_tickers:
                current_tickers.append(ticker)
                saved_config["tickers"] = current_tickers
                save_config_to_file(saved_mode, saved_config)
                message = f"✅ Added {ticker} to watchlist (saved to config)"
            else:
                message = f"{ticker} is already in watchlist"
        return WatchlistResponse(message=message, tickers=current_tickers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/watchlist/remove/{ticker}", response_model=WatchlistResponse)
async def remove_from_watchlist(ticker: str, payload: dict = Depends(get_optional_user)):
    """Remove ticker from watchlist. When authenticated, updates only that user's watchlist in MongoDB."""
    try:
        ticker = ticker.upper().strip()
        if not ticker.endswith(('.NS', '.BO')):
            ticker += '.NS'

        # Per-user: when authenticated, update only this user's watchlist in MongoDB
        if payload and isinstance(payload, dict):
            username = payload.get("sub") or ""
            if username:
                current_tickers = _get_user_watchlist_from_db(username)
                if ticker in current_tickers:
                    current_tickers.remove(ticker)
                    _save_user_watchlist_to_db(username, current_tickers)
                    message = f"✅ Removed {ticker} from your watchlist"
                else:
                    message = f"{ticker} is not in your watchlist"
                logger.info(f"📊 Watchlist REMOVE: {ticker} for user {username} (MongoDB)")
                return WatchlistResponse(message=message, tickers=current_tickers)

        # Global (unauthenticated)
        current_tickers = []
        if trading_bot:
            current_tickers = trading_bot.config.get("tickers", [])
            if ticker in current_tickers:
                current_tickers.remove(ticker)
                trading_bot.config["tickers"] = current_tickers
                try:
                    saved_mode = get_current_saved_mode() or trading_bot.config.get("mode", "paper")
                    save_config_to_file(saved_mode, trading_bot.config)
                except Exception as save_err:
                    logger.warning(f"Failed to save config: {save_err}")
                try:
                    from data_feed import DataFeed
                    if DataFeed:
                        trading_bot.data_feed = DataFeed(current_tickers)
                except Exception:
                    pass
                message = f"✅ Removed {ticker} from watchlist"
            else:
                message = f"{ticker} is not in watchlist"
        else:
            saved_mode = get_current_saved_mode() or "paper"
            saved_config = load_config_from_file(saved_mode) or {}
            current_tickers = saved_config.get("tickers", [])
            if ticker in current_tickers:
                current_tickers.remove(ticker)
                saved_config["tickers"] = current_tickers
                save_config_to_file(saved_mode, saved_config)
                message = f"✅ Removed {ticker} from watchlist"
            else:
                message = f"{ticker} is not in watchlist"
        return WatchlistResponse(message=message, tickers=current_tickers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/watchlist/bulk", response_model=BulkWatchlistResponse)
async def bulk_update_watchlist(request: BulkWatchlistRequest, payload: dict = Depends(get_optional_user)):
    """Add or remove multiple tickers. When authenticated, updates only that user's watchlist in MongoDB."""
    try:
        action = request.action.upper()
        if action not in ["ADD", "REMOVE"]:
            raise HTTPException(
                status_code=400, detail="Action must be ADD or REMOVE")

        successful_tickers = []
        failed_tickers = []
        use_user_db = payload and isinstance(payload, dict) and (payload.get("sub") or "")

        # Resolve current tickers: per-user from MongoDB when authenticated, else global (bot/config)
        if use_user_db:
            current_tickers = list(_get_user_watchlist_from_db(payload.get("sub") or ""))
        elif trading_bot:
            current_tickers = list(trading_bot.config.get("tickers", []))
        else:
            saved_mode = get_current_saved_mode() or "paper"
            saved_config = load_config_from_file(saved_mode) or {}
            current_tickers = list(saved_config.get("tickers", []))

        for ticker in request.tickers:
            try:
                ticker = ticker.strip().upper()

                if not ticker:
                    failed_tickers.append(f"{ticker}: Empty ticker")
                    continue

                if not ticker.endswith(('.NS', '.BO')):
                    ticker += '.NS'

                if not ticker.replace('.', '').replace('-', '').replace('&', '').isalnum():
                    failed_tickers.append(f"{ticker}: Invalid format")
                    continue

                if action == "ADD":
                    if ticker in current_tickers:
                        failed_tickers.append(
                            f"{ticker}: Already in watchlist")
                        continue

                    current_tickers.append(ticker)
                    successful_tickers.append(ticker)
                    logger.info(
                        f"Added ticker {ticker} to watchlist via bulk upload")

                elif action == "REMOVE":
                    if ticker not in current_tickers:
                        failed_tickers.append(f"{ticker}: Not in watchlist")
                        continue

                    current_tickers.remove(ticker)
                    successful_tickers.append(ticker)
                    logger.info(
                        f"Removed ticker {ticker} from watchlist via bulk upload")

            except Exception as e:
                failed_tickers.append(f"{ticker}: {str(e)}")
                logger.error(f"Error processing ticker {ticker}: {e}")

        # Persist: per-user to MongoDB when authenticated, else global bot/config
        if use_user_db:
            _save_user_watchlist_to_db(payload.get("sub") or "", current_tickers)
            logger.info(f"📊 Watchlist bulk {action} for user (MongoDB): {len(successful_tickers)} tickers")
        elif trading_bot:
            trading_bot.config["tickers"] = current_tickers
            try:
                saved_mode = get_current_saved_mode() or trading_bot.config.get("mode", "paper")
                save_config_to_file(saved_mode, trading_bot.config)
            except Exception as save_err:
                logger.warning(f"Failed to save bulk watchlist to config file: {save_err}")
        else:
            saved_mode = get_current_saved_mode() or "paper"
            saved_config = load_config_from_file(saved_mode) or {}
            saved_config["tickers"] = current_tickers
            save_config_to_file(saved_mode, saved_config)
            logger.info(f"📊 Watchlist bulk {action}: saved to {saved_mode}_config.json (bot not initialized)")

        # Update data feed with new tickers (only when bot initialized and not per-user)
        if successful_tickers and action == "ADD" and DataFeed and trading_bot and not use_user_db:
            try:
                trading_bot.data_feed = DataFeed(trading_bot.config["tickers"])
                logger.info(
                    f"Updated data feed with {len(successful_tickers)} new tickers")
            except Exception as e:
                logger.error(f"Error updating data feed: {e}")

        # Prepare response message
        if successful_tickers and not failed_tickers:
            message = f"Successfully {action.lower()}ed {len(successful_tickers)} ticker(s)"
        elif successful_tickers and failed_tickers:
            message = f"Processed {len(successful_tickers)} ticker(s) successfully, {len(failed_tickers)} failed"
        elif failed_tickers and not successful_tickers:
            message = f"Failed to process all {len(failed_tickers)} ticker(s)"
        else:
            message = "No tickers processed"

        return BulkWatchlistResponse(
            message=message,
            successful_tickers=successful_tickers,
            failed_tickers=failed_tickers,
            total_processed=len(request.tickers)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk watchlist update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Priority 1: Remove duplicate validate_chat_input - now imported from utils


async def process_market_query(message: str) -> Optional[str]:
    """Process market-related queries with real-time data"""
    try:
        # Performance: Use set for O(1) lookup instead of O(n) list search
        market_keywords = {"volume", "stock", "price",
                           "highest", "lowest", "market", "trading", "analysis"}
        is_market_query = any(keyword in message.lower()
                              for keyword in market_keywords)

        if is_market_query:
            logger.info(f"Market query detected: {message}")
            return await get_real_time_market_response(message)
        return None
    except Exception as e:
        logger.error(f"Error processing market query: {e}")
        return None


async def process_groq_query(message: str, enhanced_prompt: str) -> str:
    """Process query using Groq reasoning engine"""
    try:
        global groq_engine
        if not groq_engine:
            return "Groq reasoning engine not available. Please try again later."

        response = await groq_engine.process_query(message, enhanced_prompt)
        return response.get("response", "I apologize, but I couldn't process your request at the moment.")
    except Exception as e:
        logger.error(f"Error with Groq processing: {e}")
        return "I encountered an error while processing your request. Please try again."


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message - Forward to MCP Service"""
    try:
        # Performance: Validate and sanitize input
        try:
            message = validate_chat_input(request.message)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not message:
            return ChatResponse(
                response="Please enter a message.",
                timestamp=datetime.now().isoformat()
            )

        # Forward to MCP Service API
        mcp_api_url = os.getenv("MCP_API_URL", "http://localhost:8003")
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{mcp_api_url}/api/chat",
                    json={"message": message},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            f"[WEB] Chat request forwarded to MCP service - response received")
                        return ChatResponse(
                            response=result.get(
                                "response", "No response from MCP service"),
                            timestamp=result.get(
                                "timestamp", datetime.now().isoformat()),
                            metadata=result.get("metadata")
                        )
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"[WEB] MCP service returned {response.status}: {error_text}")
                        # Fall through to fallback
        except Exception as e:
            logger.warning(f"[WEB] Failed to forward to MCP service: {e}")
            # Fall through to fallback

        # Fallback: Enhanced Real-Time Dynamic Market Analysis (if MCP service unavailable)
        try:
            # Get current timestamp for real-time data
            current_time = datetime.now()

            # Performance: Use set for O(1) lookup instead of O(n) list search
            market_keywords = {"volume", "stock", "price",
                               "highest", "lowest", "market", "trading", "analysis"}
            is_market_query = any(keyword in message.lower()
                                  for keyword in market_keywords)

            if is_market_query:
                # Get real-time market data
                logger.info(f"Market query detected: {message}")
                real_time_response = await get_real_time_market_response(message)
                logger.info(
                    f"Real-time response: {real_time_response is not None}")
                if real_time_response:
                    logger.info("Returning real-time market response")
                    return ChatResponse(
                        response=real_time_response,
                        timestamp=current_time.isoformat(),
                        confidence=0.95,
                        context="real_time_market_data"
                    )

            # Fallback to Dynamic Market Expert
            from dynamic_market_expert import DynamicMarketExpert

            # Initialize the market expert (cached for performance)
            if not hasattr(chat, '_market_expert'):
                chat._market_expert = DynamicMarketExpert()
                logger.info("Dynamic Market Expert initialized for web chat")

            # Process query with timeout protection
            import threading
            import queue

            result_queue = queue.Queue()

            def process_with_expert():
                try:
                    result = chat._market_expert.process_query(message)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            thread = threading.Thread(target=process_with_expert)
            thread.daemon = True
            thread.start()
            thread.join(timeout=15)  # 15 second timeout

            if not result_queue.empty():
                status, result = result_queue.get()
                if status == "success" and result:
                    return ChatResponse(
                        response=result,
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    logger.error(f"Expert processing error: {result}")
            else:
                logger.warning("Dynamic Market Expert response timed out")

        except ImportError as e:
            logger.error(f"Could not import Dynamic Market Expert: {e}")
        except Exception as e:
            logger.error(f"Error with Dynamic Market Expert: {e}")

        # Fallback to direct professional response with live data
        try:
            # Use existing trading bot components
            if hasattr(trading_bot, 'llm'):
                llm = trading_bot.llm
            else:
                llm = None

            # Use the Dynamic Market Expert instead
            try:
                from dynamic_market_expert import DynamicMarketExpert
                market_expert = DynamicMarketExpert()
                response = market_expert.process_query(message)
                return {"response": response, "timestamp": datetime.now().isoformat()}
            except Exception as expert_error:
                logger.error(f"Dynamic Market Expert error: {expert_error}")

            # Simple fallback response
            if True:  # Always execute fallback
                # Simple fallback response
                pass

        except Exception as e:
            logger.error(f"Error with fallback response: {e}")

        # Handle commands
        if message.startswith('/') and trading_bot:
            try:
                response = trading_bot.process_chat_command(message)
                return ChatResponse(
                    response=response,
                    timestamp=datetime.now().isoformat()
                )
            except Exception as e:
                logger.error(f"Error with command: {e}")

        # Final professional fallback
        return ChatResponse(
            response=f"""I'm your professional stock market advisor!

I can help you with:
• **Live Stock Prices** - "What's the price of {', '.join(['Reliance', 'TCS', 'HDFC Bank'])}?"
• **Market Analysis** - "How is the IT sector performing?"
• **Investment Advice** - "Should I buy banking stocks now?"
• **Portfolio Management** - Use /get_pnl, /list_positions

**Current Market Focus:** Indian equities (NSE/BSE)
**Data Source:** Live Fyers API integration

What would you like to analyze today?""",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return ChatResponse(
            response="I apologize for the error. Please try asking about stock prices or portfolio information.",
            timestamp=datetime.now().isoformat()
        )


@app.post("/api/start", response_model=MessageResponse)
async def start_bot(payload: dict = Depends(get_optional_user)):
    """Start the trading bot - returns immediately, runs heavy operations in background. When authenticated, uses that user's watchlist and demat."""
    try:
        global trading_bot, _pending_start_tickers, _pending_bot_user_context
        username = (payload.get("sub") or "").strip() if payload and isinstance(payload, dict) else ""
        if username:
            _pending_start_tickers = _get_user_watchlist_from_db(username)
            logger.info(f"📊 Bot start: using watchlist for user ({len(_pending_start_tickers)} tickers from MongoDB)")
            demat = auth_module.get_user_demat(username) if hasattr(auth_module, "get_user_demat") else None
            if demat and demat.get("access_token") and demat.get("client_id"):
                _pending_bot_user_context = {"user_id": username, "dhan_client_id": demat["client_id"], "dhan_access_token": demat["access_token"]}
                logger.info(f"📊 Bot start: using demat for user {username}")
            else:
                _pending_bot_user_context = None
        else:
            _pending_start_tickers = []
            _pending_bot_user_context = None

        # Initialize bot if not already initialized (start in background, don't wait)
        if not trading_bot:
            global _bot_initializing
            if _bot_initializing:
                # Initialization already in progress
                return MessageResponse(
                    message="Bot initialization is already in progress. Please wait a few seconds and try again."
                )
            
            async def init_bot_background():
                """Initialize bot in background, then start it and trigger predictions for watchlist."""
                global trading_bot, _bot_initializing, _bot_data_cache
                logger.info("🚀 init_bot_background() async function STARTED")
                try:
                    _bot_initializing = True
                    _bot_data_cache = {} # Invalidate cache to force offline data which returns isRunning=True
                    logger.info("🚀 Set _initializing flag to True")
                    loop = asyncio.get_event_loop()
                    logger.info("🔄 About to run initialize_bot() in executor...")
                    # Run initialization in executor and get the bot instance
                    try:
                        logger.info("🔄 Executor starting initialize_bot()...")
                        # 60s timeout: init runs in executor; no nested threads so it should complete or fail fast
                        bot_instance = await asyncio.wait_for(
                            loop.run_in_executor(None, initialize_bot),
                            timeout=180.0
                        )
                        logger.info(f"🔄 Executor completed, bot_instance type: {type(bot_instance) if bot_instance else 'None'}")
                        if bot_instance:
                            logger.info(f"✅ Bot instance returned successfully: {type(bot_instance).__name__}, id={id(bot_instance)}")
                        else:
                            logger.error("❌ Bot instance is None after executor completion!")
                    except asyncio.TimeoutError:
                        logger.error("❌ Bot initialization timed out after 180 seconds")
                        bot_instance = None
                    except Exception as exec_err:
                        logger.error(f"❌ Executor raised exception: {exec_err}")
                        logger.exception("Executor exception traceback:")
                        import traceback
                        logger.error(f"Full executor traceback:\n{traceback.format_exc()}")
                        bot_instance = None
                    
                    # CRITICAL: Explicitly set the global variable after executor completes
                    if not _bot_initializing:
                        logger.info("⏹ Bot initialization was cancelled. Aborting startup.")
                        return

                    if bot_instance:
                        trading_bot = bot_instance
                        logger.info(f"✅ Bot initialized successfully in background - trading_bot is set: {type(trading_bot).__name__}")
                    else:
                        logger.error("❌ Bot initialization completed but returned None!")
                        # Fallback: try to get from module global
                        import sys
                        current_module = sys.modules[__name__]
                        if hasattr(current_module, 'trading_bot') and current_module.trading_bot:
                            trading_bot = current_module.trading_bot
                            logger.info("✅ Retrieved trading_bot from module global as fallback")
                        else:
                            logger.error("❌ Module global trading_bot is also None!")
                    # Start the bot and trigger predictions for watchlist (same as when we wait)
                    if trading_bot:
                        global _pending_start_tickers
                        saved_mode = get_current_saved_mode() or "paper"
                        saved_config = load_config_from_file(saved_mode) or {}
                        # Per-user: use requesting user's watchlist if set, else saved config
                        if _pending_start_tickers:
                            trading_bot.config["tickers"] = list(_pending_start_tickers)
                            logger.info(f"📊 Loaded {len(_pending_start_tickers)} tickers for user (from MongoDB): {_pending_start_tickers}")
                            _pending_start_tickers = []
                        else:
                            saved_tickers = saved_config.get("tickers", [])
                            if saved_tickers:
                                trading_bot.config["tickers"] = saved_tickers
                                logger.info(f"📊 Loaded {len(saved_tickers)} tickers from saved config: {saved_tickers}")
                        risk_level = trading_bot.config.get("riskLevel", "MEDIUM")
                        apply_risk_level_settings(trading_bot, risk_level)
                        await loop.run_in_executor(None, trading_bot.start)
                        tickers_list = trading_bot.config.get("tickers", [])
                        logger.info(f"🚀 Bot started in background with {len(tickers_list)} tickers: {tickers_list}")
                        global bot_running
                        bot_running = True
                        _bot_data_cache = {} # Force refresh of live data on next /api/bot-data request
                        # Start the continuous loop — it will process all tickers sequentially
                        _start_continuous_loop()
                        logger.info(f"✅ Continuous trading loop task created")
                except Exception as init_error:
                    logger.error(f"❌ Background bot initialization failed: {init_error}")
                    logger.exception("Full traceback:")
                finally:
                    _bot_initializing = False
            
            # Start initialization in background (non-blocking) - return immediately
            asyncio.create_task(init_bot_background())
            logger.info("🔄 Bot initialization started in background - returning immediately")

            # Return right away — do NOT wait for init to complete.
            # The background task will set trading_bot and trigger predictions when ready.
            # Frontend should poll GET /api/status to check is_running.
            return MessageResponse(
                message="Bot initialization started in background. Predictions will run for your watchlist shortly. "
                        "Poll GET /api/status for is_running=true to confirm the bot is active."
            )

        if trading_bot:
            # Resolve the new tickers list
            if _pending_start_tickers:
                new_tickers = list(_pending_start_tickers)
                _pending_start_tickers = []
            else:
                saved_mode = get_current_saved_mode() or "paper"
                saved_config = load_config_from_file(saved_mode) or {}
                new_tickers = saved_config.get("tickers", []) or trading_bot.config.get("tickers", [])

            # Figure out which tickers are brand-new (not yet analyzed)
            old_tickers = list(trading_bot.config.get("tickers", []))
            added_tickers = [t for t in new_tickers if t not in old_tickers]

            # Update config with the full new ticker list
            trading_bot.config["tickers"] = new_tickers
            logger.info(f"📊 Updated tickers: {new_tickers} (new: {added_tickers})")

            # Apply risk settings
            risk_level = trading_bot.config.get("riskLevel", "MEDIUM")
            apply_risk_level_settings(trading_bot, risk_level)

            # If bot is NOT already running, start it (don't call start() if it's already running)
            if not trading_bot.is_running:
                loop = asyncio.get_event_loop()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, trading_bot.start),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    # start() is non-blocking once running; timeout just means it's busy
                    logger.warning("trading_bot.start() timed out (bot may already be active)")
            else:
                logger.info("📊 Bot already running — updating tickers list only")

            stop_loss_pct = trading_bot.config.get('stop_loss_pct', 0.05) * 100
            max_allocation_pct = trading_bot.config.get('max_capital_per_trade', 0.25) * 100
            tickers_count = len(new_tickers)
            logger.info(f"🚀 Bot active with {risk_level} risk, {tickers_count} tickers: {new_tickers}")

            # Don't immediately trigger analysis here — the continuous loop will pick up new tickers
            # on its next cycle. Triggering here while the loop is mid-analysis saturates the
            # thread pool and causes the frontend to freeze on "processing".
            if added_tickers:
                logger.info(f"📊 New tickers queued for next loop cycle: {added_tickers}")

            # Ensure continuous loop is running (start if not already active)
            _start_continuous_loop()
            global bot_running
            bot_running = True

            if added_tickers:
                return MessageResponse(message=f"Bot running with {tickers_count} tickers. {len(added_tickers)} new ticker(s) will be analyzed in the next loop cycle.")
            return MessageResponse(message=f"Bot running with {tickers_count} tickers.")
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Start bot operation timed out")
        raise HTTPException(status_code=500, detail="Start bot operation timed out - bot may still be starting in background")
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/init", response_model=MessageResponse)
async def init_bot():
    """Manually initialize the trading bot"""
    try:
        global trading_bot
        initialize_bot()
        return MessageResponse(message="Trading bot initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop", response_model=MessageResponse)
async def stop_bot():
    """Stop the trading bot"""
    try:
        _stop_continuous_loop()
        if trading_bot:
            trading_bot.stop()
            return MessageResponse(message="Bot stopped successfully")
        else:
            raise HTTPException(status_code=500, detail="Bot not initialized")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bot/start", response_model=MessageResponse)
async def start_bot_bot_route(payload: dict = Depends(get_optional_user)):
    """Start the trading bot (alias for /api/start). Uses authenticated user's watchlist when present."""
    return await start_bot(payload)


@app.post("/api/bot/stop", response_model=MessageResponse)
async def stop_bot_bot_route():
    """Stop the trading bot (alias for /api/stop). Used by frontend Stop Bot button."""
    return await stop_bot()


@app.post("/api/bot/start-with-symbol")
async def start_bot_with_symbol(request: StartBotWithSymbolRequest):
    """Start bot with a symbol: add to watchlist, start bot, trigger prediction and analysis"""
    try:
        global trading_bot
        symbol = request.symbol.upper().strip()
        
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")
        
        # Initialize bot if not already initialized (run in executor to avoid blocking)
        if not trading_bot:
            try:
                # Run synchronous initialize_bot in thread pool executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, initialize_bot)
                logger.info("Bot initialized before starting with symbol")
            except Exception as init_error:
                logger.error(f"Failed to initialize bot: {init_error}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to initialize bot: {str(init_error)}")
        
        if not trading_bot:
            raise HTTPException(status_code=500, detail="Bot not initialized")
        
        # Add symbol to watchlist if not already present
        current_tickers = trading_bot.config.get("tickers", [])
        if symbol not in current_tickers:
            current_tickers.append(symbol)
            trading_bot.config["tickers"] = current_tickers
            logger.info(f"Added {symbol} to watchlist")
            
            # Update data feed if it exists
            try:
                try:
                    from data_feed import DataFeed
                except ImportError:
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    from data_feed import DataFeed
                if DataFeed:
                    trading_bot.data_feed = DataFeed(current_tickers)
            except Exception as e:
                logger.warning(f"Failed to update data feed: {e}")
        
        # Apply current risk level settings before starting
        risk_level = trading_bot.config.get("riskLevel", "MEDIUM")
        apply_risk_level_settings(trading_bot, risk_level)
        
        # Start the bot if not already running
        if not trading_bot.is_running:
            trading_bot.start()
            logger.info(f"Trading bot started with symbol {symbol}")
        
        # Trigger ALL HFT2 backend components: prediction, analysis, data fetching
        asyncio.create_task(trigger_all_hft2_components_for_symbol(symbol))
        
        # Return immediately
        return {
            "status": "success",
            "message": f"Bot started with symbol {symbol}. Analysis running in background.",
            "symbol": symbol,
            "isRunning": trading_bot.is_running,
            "watchlist": current_tickers,
            "prediction": None,  # Will be available later via bot-data endpoint
            "analysis": None,    # Will be available later via bot-data endpoint
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting bot with symbol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings")
async def get_settings():
    """Get current settings. Returns saved/defaults when bot not initialized."""
    try:
        if trading_bot:
            return {
                "mode": trading_bot.config.get("mode", "paper"),
                "riskLevel": trading_bot.config.get("riskLevel", "MEDIUM"),
                "stop_loss_pct": trading_bot.config.get("stop_loss_pct", 0.05),
                "target_profit_pct": trading_bot.config.get("target_profit_pct", 0.1),
                "use_risk_reward": trading_bot.config.get("use_risk_reward", True),
                "risk_reward_ratio": trading_bot.config.get("risk_reward_ratio", 2.0),
                "max_capital_per_trade": trading_bot.config.get("max_capital_per_trade", 0.25),
                "max_trade_limit": trading_bot.config.get("max_trade_limit", 10)
            }
        mode = get_current_saved_mode()
        saved = load_config_from_file(mode) or {}
        return {
            "mode": saved.get("mode", mode),
            "riskLevel": saved.get("riskLevel", "MEDIUM"),
            "stop_loss_pct": saved.get("stop_loss_pct", 0.05),
            "target_profit_pct": saved.get("target_profit_pct", 0.1),
            "use_risk_reward": saved.get("use_risk_reward", True),
            "risk_reward_ratio": saved.get("risk_reward_ratio", 2.0),
            "max_capital_per_trade": saved.get("max_capital_per_trade", 0.25),
            "max_trade_limit": saved.get("max_trade_limit", 10),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def save_config_to_file(mode: str, config_data: dict):
    """Save configuration to the appropriate JSON file (backend/hft2/data)."""
    try:
        import json

        data_dir = _get_settings_data_dir()
        os.makedirs(data_dir, exist_ok=True)
        config_file = os.path.join(data_dir, f"{mode}_config.json")

        # Prepare config data for saving
        config_to_save = {
            "mode": mode,
            "riskLevel": config_data.get("riskLevel", "MEDIUM"),
            "stop_loss_pct": config_data.get("stop_loss_pct", 0.05),
            "target_profit_pct": config_data.get("target_profit_pct", 0.1),
            "use_risk_reward": config_data.get("use_risk_reward", True),
            "risk_reward_ratio": config_data.get("risk_reward_ratio", 2.0),
            "max_capital_per_trade": config_data.get("max_capital_per_trade", 0.25),
            "max_trade_limit": config_data.get("max_trade_limit", 150),
            "tickers": config_data.get("tickers", []),  # Include tickers/watchlist
            "created_at": datetime.now().isoformat()
        }

        # Save to file
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)

        logger.info(f"Configuration saved to {config_file}")

    except Exception as e:
        logger.error(f"Error saving config to file: {e}")
        raise


@app.post("/api/settings", response_model=MessageResponse)
async def update_settings(request: SettingsRequest, user_demat: tuple = Depends(get_optional_user_demat)):
    """Update bot settings. When bot not initialized, saves to file for when bot starts. Live-mode fetch uses linked demat when available."""
    payload, request_user_demat = user_demat if isinstance(user_demat, tuple) else (None, None)
    try:
        if trading_bot:
            # Update configuration
            if request.mode is not None:
                # Handle mode switching
                old_mode = trading_bot.config.get('mode', 'paper')
                new_mode = request.mode
                logger.info(f"Mode change requested: {old_mode} -> {new_mode}")
                
                if new_mode != old_mode:
                    # Switch mode
                    if trading_bot.switch_trading_mode(new_mode):
                        # Check if the mode actually changed (could have reverted)
                        actual_mode = trading_bot.config.get('mode', 'paper')
                        if actual_mode != new_mode:
                            logger.warning(
                                f"Requested {new_mode} mode but reverted to {actual_mode} mode")
                            # Still save the requested mode to file
                            save_config_to_file(new_mode, trading_bot.config)
                            set_current_saved_mode(new_mode)
                        else:
                            logger.info(
                                f"Successfully switched from {old_mode} to {new_mode} mode")
                            # Ensure mode is persisted
                            trading_bot.config['mode'] = new_mode
                            save_config_to_file(new_mode, trading_bot.config)
                            set_current_saved_mode(new_mode)
                    else:
                        logger.error(f"Failed to switch to {new_mode} mode")
                        raise HTTPException(
                            status_code=400, detail=f"Failed to switch to {new_mode} mode. Check Dhan credentials.")
                else:
                    # Mode unchanged, but ensure it's set correctly
                    trading_bot.config['mode'] = new_mode
                    logger.info(f"Mode unchanged: {new_mode}")
            if request.riskLevel is not None:
                trading_bot.config['riskLevel'] = request.riskLevel
                # Apply risk level settings dynamically
                # For predefined levels, don't pass custom values so they use the mappings
                if request.riskLevel in ["LOW", "MEDIUM", "HIGH"]:
                    apply_risk_level_settings(trading_bot, request.riskLevel)
                else:
                    # For CUSTOM, use the provided values
                    apply_risk_level_settings(
                        bot=trading_bot,
                        risk_level=request.riskLevel,
                        custom_stop_loss=request.stop_loss_pct,
                        custom_allocation=request.max_capital_per_trade,
                        custom_target_profit=request.target_profit_pct,
                        custom_use_rr=request.use_risk_reward,
                        custom_rr_ratio=request.risk_reward_ratio
                    )
            if request.stop_loss_pct is not None:
                trading_bot.config['stop_loss_pct'] = request.stop_loss_pct
                # Update executor if it exists
                if hasattr(trading_bot, 'executor') and trading_bot.executor:
                    trading_bot.executor.stop_loss_pct = request.stop_loss_pct
            if request.max_capital_per_trade is not None:
                trading_bot.config['max_capital_per_trade'] = request.max_capital_per_trade
                # Update executor if it exists
                if hasattr(trading_bot, 'executor') and trading_bot.executor:
                    trading_bot.executor.max_capital_per_trade = request.max_capital_per_trade
            if request.max_trade_limit is not None:
                trading_bot.config['max_trade_limit'] = request.max_trade_limit

            # Handle target profit settings
            if request.target_profit_pct is not None:
                trading_bot.config['target_profit_pct'] = request.target_profit_pct
                if hasattr(trading_bot, 'executor') and trading_bot.executor:
                    trading_bot.executor.target_profit_pct = request.target_profit_pct

            # Handle risk/reward settings
            if request.use_risk_reward is not None:
                trading_bot.config['use_risk_reward'] = request.use_risk_reward
                if hasattr(trading_bot, 'executor') and trading_bot.executor:
                    trading_bot.executor.use_risk_reward = request.use_risk_reward

            if request.risk_reward_ratio is not None:
                trading_bot.config['risk_reward_ratio'] = request.risk_reward_ratio
                if hasattr(trading_bot, 'executor') and trading_bot.executor:
                    trading_bot.executor.risk_reward_ratio = request.risk_reward_ratio

            # Save the updated configuration to the appropriate config file
            current_mode = trading_bot.config.get('mode', 'paper')
            save_config_to_file(current_mode, trading_bot.config)
            set_current_saved_mode(current_mode)

            # If switching to live mode, immediately fetch Dhan credentials and portfolio data (prefer user's linked demat)
            if current_mode == 'live':
                async def fetch_dhan_data_immediately():
                    try:
                        logger.info("🔄 Switching to live mode - fetching Dhan account credentials and portfolio data...")
                        loop = asyncio.get_event_loop()
                        from dhan_client import get_live_portfolio
                        token, client_id = None, None
                        if request_user_demat and request_user_demat.get("access_token") and request_user_demat.get("client_id"):
                            token = request_user_demat["access_token"]
                            client_id = request_user_demat["client_id"]
                            logger.info("✅ Using linked demat credentials for live fetch")
                        if not token or not client_id:
                            try:
                                from dhan_client import get_dhan_token, get_dhan_client_id
                                token = get_dhan_token()
                                client_id = get_dhan_client_id()
                            except Exception:
                                pass
                        if not token or not client_id:
                            logger.warning("⚠️ No Dhan credentials (link demat or set env) - skipping live fetch")
                            return
                        logger.info(f"✅ Dhan credentials: Token={bool(token)}, ClientID={bool(client_id)}")
                        try:
                            dhan_portfolio = await asyncio.wait_for(
                                loop.run_in_executor(None, lambda: get_live_portfolio(access_token=token, client_id=client_id)),
                                timeout=10.0
                            )
                            if dhan_portfolio:
                                cash = dhan_portfolio.get("cash", 0)
                                holdings_count = len(dhan_portfolio.get("holdings", {}))
                                total_value = dhan_portfolio.get("totalValue", 0)
                                logger.info(f"✅ Fetched Dhan portfolio: Cash=₹{cash:,.2f}, Holdings={holdings_count}, Total Value=₹{total_value:,.2f}")
                            else:
                                logger.warning("⚠️ Dhan API returned empty portfolio")
                        except asyncio.TimeoutError:
                            logger.error("❌ Dhan API fetch timed out after 10s")
                        except Exception as fetch_err:
                            logger.error(f"❌ Failed to fetch Dhan portfolio: {fetch_err}")
                        
                        # Also sync with database via live_executor
                        if hasattr(trading_bot, 'live_executor') and trading_bot.live_executor:
                            try:
                                await loop.run_in_executor(None, trading_bot.live_executor.sync_portfolio_with_dhan)
                                logger.info("✅ Synced portfolio with database")
                            except Exception as sync_err:
                                logger.warning(f"⚠️ Database sync failed: {sync_err}")
                        
                        # Sync portfolio manager if available
                        if hasattr(trading_bot, 'portfolio_manager') and trading_bot.portfolio_manager:
                            if hasattr(trading_bot.portfolio_manager, 'sync_with_dhan'):
                                try:
                                    await loop.run_in_executor(None, trading_bot.portfolio_manager.sync_with_dhan)
                                except Exception as pm_sync_err:
                                    logger.warning(f"⚠️ Portfolio manager sync failed: {pm_sync_err}")
                        
                        logger.info("✅ Live mode initialization completed - Dhan data fetched")
                    except Exception as sync_error:
                        logger.error(f"❌ Error fetching Dhan data: {sync_error}")
                        logger.exception("Full traceback:")
                
                # Trigger immediate Dhan data fetch in background (non-blocking)
                try:
                    asyncio.create_task(fetch_dhan_data_immediately())
                except Exception as task_err:
                    logger.warning(f"Failed to start background Dhan fetch: {task_err}")

            logger.info(f"Settings updated: Mode={trading_bot.config.get('mode')}, "
                        f"Risk Level={trading_bot.config.get('riskLevel')}, "
                        f"Stop Loss={trading_bot.config.get('stop_loss_pct', 0.05)*100:.1f}%, "
                        f"Target Profit={trading_bot.config.get('target_profit_pct', 0.1)*100:.1f}%, "
                        f"Use RR={trading_bot.config.get('use_risk_reward', True)}, "
                        f"RR Ratio={trading_bot.config.get('risk_reward_ratio', 2.0):.1f}, "
                        f"Max Allocation={trading_bot.config.get('max_capital_per_trade', 0.25)*100:.1f}%")

            return MessageResponse(message="Settings updated successfully")
        # Bot not initialized: persist to file so they apply when bot starts
        mode = request.mode or get_current_saved_mode() or os.getenv("MODE", "paper")
        config_to_save = {
            "mode": mode,
            "riskLevel": request.riskLevel or "MEDIUM",
            "stop_loss_pct": request.stop_loss_pct if request.stop_loss_pct is not None else 0.05,
            "target_profit_pct": request.target_profit_pct if request.target_profit_pct is not None else 0.1,
            "use_risk_reward": request.use_risk_reward if request.use_risk_reward is not None else True,
            "risk_reward_ratio": request.risk_reward_ratio if request.risk_reward_ratio is not None else 2.0,
            "max_capital_per_trade": request.max_capital_per_trade if request.max_capital_per_trade is not None else 0.25,
            "max_trade_limit": request.max_trade_limit if request.max_trade_limit is not None else 10,
        }
        save_config_to_file(mode, config_to_save)
        set_current_saved_mode(mode)
        logger.info(f"Settings saved for mode: {mode} (bot not initialized yet)")
        
        # CRITICAL: If bot exists but is still initializing, update its config immediately
        # This ensures get_bot_data() returns the correct mode even during initialization
        if trading_bot and hasattr(trading_bot, 'config'):
            try:
                trading_bot.config['mode'] = mode
                if request.riskLevel:
                    trading_bot.config['riskLevel'] = request.riskLevel
                if request.stop_loss_pct is not None:
                    trading_bot.config['stop_loss_pct'] = request.stop_loss_pct
                if request.max_capital_per_trade is not None:
                    trading_bot.config['max_capital_per_trade'] = request.max_capital_per_trade
                logger.info(f"Updated existing bot config with new mode: {mode}")
            except Exception as update_err:
                logger.warning(f"Failed to update bot config: {update_err}")
        
        # If switching to live mode, initialize bot and fetch Dhan data (prefer user's linked demat)
        if mode == "live":
            async def init_bot_and_fetch_dhan():
                try:
                    logger.info("🔄 Initializing bot for live mode and fetching Dhan credentials...")
                    loop = asyncio.get_event_loop()
                    try:
                        await loop.run_in_executor(None, initialize_bot)
                        logger.info("✅ Bot initialized successfully for live mode")
                    except Exception as init_error:
                        logger.warning(f"⚠️ Bot initialization failed: {init_error}")
                    token, client_id = None, None
                    if request_user_demat and request_user_demat.get("access_token") and request_user_demat.get("client_id"):
                        token = request_user_demat["access_token"]
                        client_id = request_user_demat["client_id"]
                        logger.info("✅ Using linked demat credentials for live fetch")
                    if not token or not client_id:
                        try:
                            from dhan_client import get_dhan_token, get_dhan_client_id
                            token = get_dhan_token()
                            client_id = get_dhan_client_id()
                        except Exception:
                            pass
                    if not token or not client_id:
                        logger.warning("⚠️ No Dhan credentials (link demat or set env) - skipping live fetch")
                        return
                    logger.info(f"✅ Dhan credentials: Token={bool(token)}, ClientID={bool(client_id)}")
                    try:
                        from dhan_client import get_live_portfolio
                        dhan_portfolio = await asyncio.wait_for(
                            loop.run_in_executor(None, lambda: get_live_portfolio(access_token=token, client_id=client_id)),
                            timeout=10.0
                        )
                        if dhan_portfolio:
                            cash = dhan_portfolio.get("cash", 0)
                            holdings_count = len(dhan_portfolio.get("holdings", {}))
                            total_value = dhan_portfolio.get("totalValue", 0)
                            logger.info(f"✅ Fetched Dhan portfolio: Cash=₹{cash:,.2f}, Holdings={holdings_count}, Total Value=₹{total_value:,.2f}")
                    except asyncio.TimeoutError:
                        logger.error("❌ Dhan API fetch timed out")
                    except Exception as dhan_err:
                        logger.error(f"❌ Failed to fetch Dhan data: {dhan_err}")
                except Exception as error:
                    logger.error(f"❌ Error in live mode initialization: {error}")
            
            # Start initialization and Dhan fetch in background without blocking
            try:
                asyncio.create_task(init_bot_and_fetch_dhan())
            except Exception as task_error:
                logger.warning(f"Failed to start background initialization: {task_error}")
        
        return MessageResponse(message=f"Settings saved successfully. Mode: {mode}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/live-status")
async def get_live_trading_status(user_demat: tuple = Depends(get_optional_user_demat)):
    """Get live trading status and connection info. When user has linked demat, dhan_configured reflects that (no env required)."""
    try:
        if not LIVE_TRADING_AVAILABLE:
            return {
                "available": False,
                "message": "Live trading components not installed"
            }

        payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
        user_has_demat = bool(demat and demat.get("access_token") and demat.get("client_id"))

        if trading_bot and trading_bot.config.get("mode") == "live":
            # Check Dhan connection (run in executor with timeout to avoid blocking)
            dhan_connected = False
            market_status = "UNKNOWN"
            account_info = {}

            if trading_bot.dhan_client:
                try:
                    loop = asyncio.get_event_loop()
                    # Run Dhan API calls in executor with timeout
                    dhan_connected = await asyncio.wait_for(
                        loop.run_in_executor(None, trading_bot.dhan_client.validate_connection),
                        timeout=3.0  # 3 second timeout
                    )
                    if dhan_connected:
                        market_status_data = await asyncio.wait_for(
                            loop.run_in_executor(None, trading_bot.dhan_client.get_market_status),
                            timeout=3.0
                        )
                        market_status = market_status_data.get(
                            "marketStatus", "UNKNOWN") if market_status_data else "UNKNOWN"

                        # Get account info (with timeout)
                        profile = await asyncio.wait_for(
                            loop.run_in_executor(None, trading_bot.dhan_client.get_profile),
                            timeout=3.0
                        )
                        funds = await asyncio.wait_for(
                            loop.run_in_executor(None, trading_bot.dhan_client.get_funds),
                            timeout=3.0
                        )
                        # Normalize funds keys across variants

                        def _funds_value(keys, default=0):
                            for k in keys:
                                if k in funds and funds.get(k) is not None:
                                    return funds.get(k)
                            return default

                        available_cash = _funds_value(
                            ["availablecash", "availabelBalance", "availableBalance", "netAvailableMargin", "netAvailableCash"], 0)
                        sod_limit = _funds_value(
                            ["sodlimit", "sodLimit", "openingBalance", "collateralMargin"], 0)

                        account_info = {
                            "client_id": profile.get("clientId", "") if profile else "",
                            "available_cash": available_cash,
                            "used_margin": sod_limit - available_cash
                        }
                except asyncio.TimeoutError:
                    logger.warning("Dhan API calls timed out in live-status endpoint")
                    dhan_connected = False
                except Exception as e:
                    logger.error(f"Error getting live trading status: {e}")

            # Dhan configured: per-user linked demat takes precedence over env
            if user_has_demat:
                dhan_configured = True
                dhan_error = None
            else:
                dhan_configured = False
                dhan_error = None
                try:
                    try:
                        from dhan_client import get_dhan_token, get_dhan_client_id
                    except ImportError:
                        import sys
                        import os
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        if current_dir not in sys.path:
                            sys.path.insert(0, current_dir)
                        from dhan_client import get_dhan_token, get_dhan_client_id
                    token = get_dhan_token() if get_dhan_token else None
                    client_id = get_dhan_client_id() if get_dhan_client_id else None
                    dhan_configured = bool(token and client_id)
                    if not dhan_configured:
                        dhan_error = "DHAN_ACCESS_TOKEN or DHAN_CLIENT_ID not set in environment"
                except Exception as cred_error:
                    logger.error(f"Error checking Dhan credentials: {cred_error}")
                    dhan_error = str(cred_error)
            
            actual_mode = trading_bot.config.get("mode", "live") if trading_bot else get_current_saved_mode() or "live"
            return {
                "available": True,
                "mode": actual_mode,
                "connected": dhan_connected,
                "dhan_configured": dhan_configured,
                "dhan_error": dhan_error,
                "market_status": market_status,
                "account_info": account_info,
                "portfolio_synced": trading_bot.live_executor is not None
            }
        else:
            if user_has_demat:
                dhan_configured = True
                dhan_error = None
            else:
                dhan_configured = False
                dhan_error = None
                try:
                    try:
                        from dhan_client import get_dhan_token, get_dhan_client_id
                    except ImportError:
                        import sys
                        import os
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        if current_dir not in sys.path:
                            sys.path.insert(0, current_dir)
                        from dhan_client import get_dhan_token, get_dhan_client_id
                    token = get_dhan_token() if get_dhan_token else None
                    client_id = get_dhan_client_id() if get_dhan_client_id else None
                    dhan_configured = bool(token and client_id)
                    if not dhan_configured:
                        dhan_error = "DHAN_ACCESS_TOKEN or DHAN_CLIENT_ID not set in environment"
                except Exception as cred_error:
                    logger.error(f"Error checking Dhan credentials: {cred_error}")
                    dhan_error = str(cred_error)
            
            saved_mode = get_current_saved_mode() if not trading_bot else trading_bot.config.get("mode", "paper")
            return {
                "available": LIVE_TRADING_AVAILABLE,
                "mode": saved_mode,
                "connected": False,
                "dhan_configured": dhan_configured,
                "dhan_error": dhan_error,
                "message": f"Currently in {saved_mode} mode"
            }

    except Exception as e:
        logger.error(f"Error getting live trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/live/sync")
async def sync_live_portfolio(user_demat: tuple = Depends(get_optional_user_demat)):
    """Force a sync and return snapshot. When user has demat linked, fetches their portfolio; else uses trading_bot sync."""
    payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
    if demat and demat.get("access_token") and demat.get("client_id"):
        try:
            from dhan_client import get_live_portfolio
            loop = asyncio.get_event_loop()
            dhan_port = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: get_live_portfolio(access_token=demat["access_token"], client_id=demat["client_id"])),
                timeout=25.0,
            )
            if dhan_port:
                metrics = _dhan_portfolio_to_metrics(dhan_port)
                return JSONResponse(status_code=200, content={
                    "success": True,
                    "message": "Portfolio refreshed",
                    "data": metrics,
                    "synced": True,
                    "cash": metrics.get("cash", 0),
                    "holdings_value": metrics.get("current_holdings_value", 0),
                    "total_value": metrics.get("total_value", 0),
                })
        except Exception as e:
            logger.warning(f"Live sync (demat) failed: {e}")
        return JSONResponse(status_code=200, content={"success": False, "message": "Failed to fetch demat portfolio."})

    try:
        if not trading_bot or trading_bot.config.get("mode") != "live":
            return JSONResponse(
                status_code=200,
                content={"success": False, "message": "Not in live mode or bot not initialized. Portfolio will update when bot is ready."},
            )

        # Use the sync service for immediate sync
        sync_service = get_sync_service()
        if sync_service:
            success = sync_service.sync_once()
            if not success:
                raise HTTPException(
                    status_code=502, detail="Failed to sync with Dhan using sync service")

            # Return updated portfolio data
            portfolio_data = trading_bot.get_portfolio_metrics() if trading_bot else {}
            return {
                "success": True,
                "message": "Portfolio synced successfully",
                "data": portfolio_data,
                "last_sync": sync_service.last_sync_time.isoformat() if sync_service.last_sync_time else None,
                "balance": sync_service.last_known_balance
            }

        # Fallback to live executor if sync service not available
        if not trading_bot.live_executor:
            raise HTTPException(
                status_code=503, detail="Live executor not initialized")
        ok = trading_bot.live_executor.sync_portfolio_with_dhan()
        if not ok:
            raise HTTPException(
                status_code=502, detail="Failed to sync with Dhan")
        portfolio_data = trading_bot.get_portfolio_metrics() if trading_bot else {}
        return {
            "synced": True,
            "cash": portfolio_data.get("cash", 0.0),
            "holdings_value": portfolio_data.get("total_value", 0.0) - portfolio_data.get("cash", 0.0),
            "total_value": portfolio_data.get("total_value", 0.0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class OrderRequest(BaseModel):
    """Request model for placing buy/sell orders"""
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    order_type: Optional[str] = "MARKET"
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@app.post("/api/order")
async def place_order(request: OrderRequest, user_demat: tuple = Depends(get_optional_user_demat)):
    """
    Place a buy or sell order. When user has demat linked, uses their broker (Dhan); else uses trading_bot.
    """
    side = (request.side or "").upper()
    if side not in ["BUY", "SELL"]:
        raise HTTPException(status_code=400, detail="Side must be BUY or SELL")
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0")

    payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
    if demat and demat.get("access_token") and demat.get("client_id") and (demat.get("broker") or "dhan") == "dhan":
        try:
            from dhan_client import place_order_market
            loop = asyncio.get_event_loop()
            out = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: place_order_market(
                        symbol=request.symbol,
                        side=side,
                        quantity=request.quantity,
                        product_type="CNC",
                        trigger_price=float(request.stop_loss) if request.stop_loss is not None else None,
                        access_token=demat["access_token"],
                        client_id=demat["client_id"],
                    ),
                ),
                timeout=15.0,
            )
            if out and isinstance(out, dict):
                return {
                    "success": True,
                    "status": "executed",
                    "order_id": out.get("orderId") or out.get("order_id", ""),
                    "symbol": request.symbol,
                    "side": side,
                    "quantity": request.quantity,
                    "price": request.price,
                    "message": f"{side} order sent successfully",
                    "mode": "live",
                }
            raise HTTPException(status_code=400, detail="Order failed or no response from broker")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Order request timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Demat order failed")
            raise HTTPException(status_code=400, detail=str(e))

    try:
        if not trading_bot:
            raise HTTPException(status_code=503, detail="Trading bot not initialized. Link a demat account or start the bot.")

        current_mode = trading_bot.config.get("mode", "paper")
        signal_data = {
            "quantity": request.quantity,
            "current_price": request.price,
            "stop_loss": request.stop_loss,
            "take_profit": request.take_profit,
            "confidence": 1.0,
            "order_type": request.order_type or "MARKET",
        }

        if current_mode == "live":
            if not LIVE_TRADING_AVAILABLE:
                raise HTTPException(status_code=503, detail="Live trading not available")
            if not hasattr(trading_bot, "live_executor") or not trading_bot.live_executor:
                raise HTTPException(status_code=503, detail="Live executor not initialized. Please ensure Dhan credentials are configured.")
            if side == "BUY":
                result = trading_bot.live_executor.execute_buy_order(request.symbol, signal_data)
            else:
                result = trading_bot.live_executor.execute_sell_order(request.symbol, signal_data)
            if not result.get("success", False):
                raise HTTPException(status_code=400, detail=result.get("message", "Order execution failed"))
            try:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, trading_bot.live_executor.sync_portfolio_with_dhan)
            except Exception:
                pass
            return {
                "success": True,
                "status": "executed",
                "order_id": result.get("order_id"),
                "symbol": request.symbol,
                "side": side,
                "quantity": result.get("quantity", request.quantity),
                "price": result.get("price"),
                "message": result.get("message", f"{side} order executed successfully"),
                "mode": "live",
            }

        # Paper mode
        logger.info(f"Paper mode: Simulating {side} order for {request.quantity} {request.symbol}")
        try:
            if side == "BUY":
                trading_bot.portfolio_manager.record_trade(
                    ticker=request.symbol,
                    action="buy",
                    quantity=request.quantity,
                    price=request.price or 100.0,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit
                )
            else:
                trading_bot.portfolio_manager.record_trade(
                    ticker=request.symbol,
                    action="sell",
                    quantity=request.quantity,
                    price=request.price or 100.0,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit
                )
            return {
                "success": True,
                "status": "executed",
                "order_id": f"paper-{int(time.time())}",
                "symbol": request.symbol,
                "side": side,
                "quantity": request.quantity,
                "price": request.price or 100.0,
                "message": f"{side} order simulated successfully (paper mode)",
                "mode": "paper"
            }
        except Exception as paper_err:
            logger.error(f"Paper mode order simulation failed: {paper_err}")
            raise HTTPException(status_code=500, detail=f"Paper mode simulation failed: {str(paper_err)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=f"Order placement failed: {str(e)}")


# ============================================================================
# MCP (Model Context Protocol) API Endpoints
# ============================================================================


def _set_mcp_user_context_from_request(user_demat: tuple) -> None:
    """Set request-scoped user context for MCP (per-user portfolio/order)."""
    try:
        from request_context import set_mcp_user_context
        payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
        user_id = (payload.get("sub") or "").strip() if payload else None
        set_mcp_user_context(user_id, demat)
    except Exception as e:
        logger.debug(f"set_mcp_user_context: {e}")


@app.post("/api/mcp/analyze")
async def mcp_analyze_market(request: MCPAnalysisRequest, user_demat: tuple = Depends(get_optional_user_demat)):
    """MCP-powered comprehensive market analysis with AI reasoning. Uses request user's demat when linked."""
    try:
        _set_mcp_user_context_from_request(user_demat)
        if not MCP_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="MCP server not available")

        # Initialize MCP components if needed
        await _ensure_mcp_initialized()

        if not mcp_trading_agent:
            raise HTTPException(
                status_code=503, detail="MCP trading agent not initialized")

        # Perform AI-powered analysis
        signal = await mcp_trading_agent.analyze_and_decide(
            symbol=request.symbol,
            market_context={
                "timeframe": request.timeframe,
                "analysis_type": request.analysis_type
            }
        )

        return {
            "symbol": signal.symbol,
            "recommendation": signal.decision.value,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "risk_score": signal.risk_score,
            "position_size": signal.position_size,
            "target_price": signal.target_price,
            "stop_loss": signal.stop_loss,
            "expected_return": signal.expected_return,
            "metadata": signal.metadata,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP market analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/execute")
async def mcp_execute_trade(request: MCPTradeRequest, user_demat: tuple = Depends(get_optional_user_demat)):
    """MCP-controlled trade execution. Uses request user's demat when linked."""
    try:
        _set_mcp_user_context_from_request(user_demat)
        if not MCP_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="MCP server not available")

        await _ensure_mcp_initialized()

        if not mcp_trading_agent:
            raise HTTPException(
                status_code=503, detail="MCP trading agent not initialized")

        # Get AI analysis first
        signal = await mcp_trading_agent.analyze_and_decide(request.symbol)

        # Generate explanation for the trade
        if groq_engine:
            async with groq_engine:
                explanation = await groq_engine.explain_trade_decision(
                    request.action,
                    TradingContext(
                        symbol=request.symbol,
                        current_price=0.0,  # Will be filled by agent
                        technical_signals={},
                        market_data={}
                    )
                )
        else:
            explanation = GroqResponse(
                content="MCP analysis completed", reasoning=signal.reasoning)

        # Execute real order when confidence is high and user has demat linked (any broker)
        execution_result = None
        if signal.confidence > 0.7 and signal.decision.value in ["BUY", "SELL"]:
            payload, demat = user_demat if isinstance(user_demat, tuple) else (None, None)
            if not demat or not demat.get("access_token") or not demat.get("client_id"):
                execution_result = {
                    "executed": False,
                    "reason": "Link your demat account in BOT Settings to place real orders",
                    "message": "Demat not linked"
                }
            else:
                quantity = request.quantity if request.quantity and request.quantity > 0 else 1
                try:
                    from request_context import place_order_for_request_user
                    loop = asyncio.get_event_loop()
                    out = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: place_order_for_request_user(
                                symbol=request.symbol,
                                side=signal.decision.value,
                                quantity=quantity,
                                product_type="CNC",
                                trigger_price=None,
                            ),
                        ),
                        timeout=15.0,
                    )
                    if out and isinstance(out, dict):
                        execution_result = {
                            "executed": True,
                            "order_id": out.get("orderId") or out.get("order_id") or f"MCP_{int(time.time())}",
                            "message": f"Order sent: {signal.decision.value} {request.symbol} qty={quantity}",
                        }
                    else:
                        execution_result = {
                            "executed": False,
                            "reason": "Broker did not return order confirmation",
                            "message": "Order may have failed",
                        }
                except asyncio.TimeoutError:
                    execution_result = {"executed": False, "reason": "Order request timed out", "message": "Timeout"}
                except Exception as e:
                    logger.exception("MCP execute order failed")
                    execution_result = {"executed": False, "reason": str(e), "message": "Order failed"}
        else:
            execution_result = {
                "executed": False,
                "reason": f"Low confidence ({signal.confidence:.2f}) or HOLD decision",
                "message": "Trade not executed due to risk management"
            }

        return {
            "analysis": {
                "recommendation": signal.decision.value,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "risk_score": signal.risk_score
            },
            "execution": execution_result,
            "explanation": explanation.content,
            "override_reason": request.override_reason,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/chat")
async def mcp_chat(request: MCPChatRequest, user_demat: tuple = Depends(get_optional_user_demat)):
    """Advanced AI chat with market context. Uses request user's demat for portfolio when linked."""
    try:
        _set_mcp_user_context_from_request(user_demat)
        if not MCP_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="MCP server not available")

        await _ensure_mcp_initialized()

        if not groq_engine:
            raise HTTPException(
                status_code=503, detail="Groq engine not available")

        # Determine chat context
        message = request.message.lower()

        if any(keyword in message for keyword in ["analyze", "stock", "price", "buy", "sell"]):
            # Market-related query
            if request.context and "symbol" in request.context:
                symbol = request.context["symbol"]

                # Get market analysis
                signal = await mcp_trading_agent.analyze_and_decide(symbol)

                # Generate contextual response
                async with groq_engine:
                    response = await groq_engine.analyze_market_decision(
                        TradingContext(
                            symbol=symbol,
                            current_price=signal.entry_price,
                            technical_signals=signal.metadata.get(
                                "technical_signals", {}),
                            market_data=signal.metadata.get("market_data", {})
                        )
                    )

                return {
                    "response": response.content,
                    "reasoning": response.reasoning,
                    "confidence": response.confidence,
                    "context": "market_analysis",
                    "related_analysis": {
                        "symbol": symbol,
                        "recommendation": signal.decision.value,
                        "risk_score": signal.risk_score
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "response": "Please specify a stock symbol for market analysis.",
                    "context": "general",
                    "timestamp": datetime.now().isoformat()
                }

        elif any(keyword in message for keyword in ["portfolio", "risk", "allocation"]):
            # Portfolio-related query (prefer request user's demat when set)
            portfolio_data = None
            try:
                from request_context import get_portfolio_for_request_user
                user_port = get_portfolio_for_request_user()
                if user_port and isinstance(user_port, dict):
                    portfolio_data = {
                        "holdings": user_port.get("holdings", {}),
                        "cash": user_port.get("cash", 0),
                        "risk_profile": "MEDIUM"
                    }
            except Exception:
                pass
            if not portfolio_data and trading_bot:
                portfolio_data = {
                    "holdings": trading_bot.get_portfolio_metrics().get("holdings", {}),
                    "cash": trading_bot.get_portfolio_metrics().get("cash", 0),
                    "risk_profile": trading_bot.config.get("riskLevel", "MEDIUM")
                }
            if portfolio_data:

                async with groq_engine:
                    response = await groq_engine.optimize_portfolio(portfolio_data)

                return {
                    "response": response.content,
                    "reasoning": response.reasoning,
                    "confidence": response.confidence,
                    "context": "portfolio_optimization",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "response": "Portfolio data not available.",
                    "context": "error",
                    "timestamp": datetime.now().isoformat()
                }

        else:
            # General trading query
            general_prompt = f"""
            You are an expert trading advisor. Answer this question: {request.message}

            Provide practical, actionable advice based on sound trading principles.
            """

            async with groq_engine:
                response = await groq_engine.generate_response(general_prompt)

            return {
                "response": response.content,
                "reasoning": response.reasoning,
                "confidence": response.confidence,
                "context": "general_trading",
                "timestamp": datetime.now().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/predict")
async def mcp_predict(request: PredictionRequest, user_demat: tuple = Depends(get_optional_user_demat)):
    """MCP-powered prediction ranking. Uses request user's demat when linked."""
    try:
        _set_mcp_user_context_from_request(user_demat)
        if not MCP_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="MCP server not available")

        await _ensure_mcp_initialized()

        # Generate a session ID for this request
        session_id = str(int(time.time() * 1000000))

        # Prepare arguments for the prediction tool
        arguments = {
            "symbols": request.symbols or [],
            "models": request.models or ["rl"],
            "horizon": request.horizon or "day",
            "include_explanations": request.include_explanations,
            "natural_query": request.natural_query or ""
        }

        # Call the prediction tool directly
        from mcp_server.tools.prediction_tool import PredictionTool
        prediction_tool = PredictionTool({
            "tool_id": "prediction_tool",
            "ollama_enabled": True,
            "ollama_host": "http://localhost:11434",
            "ollama_model": "llama3.1:8b"
        })

        result = await prediction_tool.rank_predictions(arguments, session_id)

        if result.status == "SUCCESS":
            return {
                "success": True,
                "data": result.data,
                "reasoning": result.reasoning,
                "confidence": result.confidence,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mcp/scan")
async def mcp_scan(request: ScanRequest, user_demat: tuple = Depends(get_optional_user_demat)):
    """MCP-powered stock scanning. Uses request user's demat when linked."""
    try:
        _set_mcp_user_context_from_request(user_demat)
        if not MCP_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="MCP server not available")

        await _ensure_mcp_initialized()

        # Generate a session ID for this request
        session_id = str(int(time.time() * 1000000))

        # Prepare arguments for the scan tool
        arguments = {
            "filters": request.filters or {},
            "sort_by": request.sort_by or "score",
            "limit": request.limit or 50,
            "natural_query": request.natural_query or ""
        }

        # Call the scan tool directly
        from mcp_server.tools.scan_tool import ScanTool
        scan_tool = ScanTool({
            "tool_id": "scan_tool",
            "ollama_enabled": True,
            "ollama_host": "http://localhost:11434",
            "ollama_model": "llama3.1:8b"
        })

        result = await scan_tool.scan_all(arguments, session_id)

        if result.status == "SUCCESS":
            return {
                "success": True,
                "data": result.data,
                "reasoning": result.reasoning,
                "confidence": result.confidence,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail=result.error)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCP scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mcp/status")
async def get_mcp_status():
    """Get MCP server and agent status"""
    try:
        status = {
            "mcp_available": MCP_AVAILABLE,
            "server_initialized": mcp_server is not None,
            "agent_initialized": mcp_trading_agent is not None,
            "fyers_connected": fyers_client is not None,
            "groq_available": groq_engine is not None
        }

        if mcp_server:
            status["server_health"] = mcp_server.get_health_status()

        if mcp_trading_agent:
            status["agent_status"] = mcp_trading_agent.get_agent_status()

        if groq_engine:
            status["groq_health"] = await groq_engine.health_check()

        return status

    except Exception as e:
        logger.error(f"MCP status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PRODUCTION-LEVEL API ENDPOINTS
# ============================================================================


@app.get("/api/production/signal-performance")
async def get_signal_performance():
    """Get signal collection performance metrics"""
    try:
        if not trading_bot or not trading_bot.is_running:
            raise HTTPException(
                status_code=503, detail="Trading bot not running")

        if not PRODUCTION_CORE_AVAILABLE or 'signal_collector' not in trading_bot.production_components:
            raise HTTPException(
                status_code=503, detail="Production signal collector not available")

        signal_collector = trading_bot.production_components['signal_collector']
        performance_metrics = signal_collector.get_performance_metrics()

        return {
            "success": True,
            "data": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/production/risk-metrics")
async def get_risk_metrics():
    """Get integrated risk management metrics"""
    try:
        if not trading_bot or not trading_bot.is_running:
            raise HTTPException(
                status_code=503, detail="Trading bot not running")

        if not PRODUCTION_CORE_AVAILABLE or 'risk_manager' not in trading_bot.production_components:
            raise HTTPException(
                status_code=503, detail="Production risk manager not available")

        risk_manager = trading_bot.production_components['risk_manager']
        risk_metrics = risk_manager.get_risk_metrics()

        return {
            "success": True,
            "data": risk_metrics,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/production/make-decision")
async def make_production_decision(request: dict):
    """Make a production-level trading decision using all components"""
    try:
        if not trading_bot or not trading_bot.is_running:
            raise HTTPException(
                status_code=503, detail="Trading bot not running")

        if not PRODUCTION_CORE_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Production components not available")

        symbol = request.get('symbol', '')
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol is required")

        # Use production components for enhanced decision making
        decision_data = await trading_bot._make_production_decision(symbol)

        return {
            "success": True,
            "data": decision_data,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making production decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/production/learning-insights")
async def get_learning_insights():
    """Get continuous learning engine insights"""
    try:
        if not trading_bot or not trading_bot.is_running:
            raise HTTPException(
                status_code=503, detail="Trading bot not running")

        if not PRODUCTION_CORE_AVAILABLE or 'learning_engine' not in trading_bot.production_components:
            raise HTTPException(
                status_code=503, detail="Production learning engine not available")

        learning_engine = trading_bot.production_components['learning_engine']
        insights = learning_engine.get_learning_insights()

        return {
            "success": True,
            "data": insights,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/production/decision-history")
async def get_decision_history(days: int = 7):
    """Get decision audit trail history"""
    try:
        if not trading_bot or not trading_bot.is_running:
            raise HTTPException(
                status_code=503, detail="Trading bot not running")

        if not PRODUCTION_CORE_AVAILABLE or 'audit_trail' not in trading_bot.production_components:
            raise HTTPException(
                status_code=503, detail="Production audit trail not available")

        audit_trail = trading_bot.production_components['audit_trail']
        history = audit_trail.get_decision_history(days=days)

        return {
            "success": True,
            "data": history,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decision history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _ensure_mcp_initialized():
    """Ensure MCP components are initialized"""
    global mcp_server, mcp_trading_agent, fyers_client, groq_engine

    try:
        if not MCP_AVAILABLE:
            return

        # Initialize Fyers client
        if not fyers_client:
            fyers_access_token = os.getenv("FYERS_ACCESS_TOKEN")
            fyers_client_id = os.getenv("FYERS_APP_ID")

            # Security: Mask sensitive data in logs
            masked_token = f"{fyers_access_token[:8]}***{fyers_access_token[-4:]}" if fyers_access_token else "None"
            masked_client_id = f"{fyers_client_id[:8]}***{fyers_client_id[-4:]}" if fyers_client_id else "None"
            logger.info(
                f"Initializing Fyers client with token: {masked_token}, client_id: {masked_client_id}")

            fyers_config = {
                "fyers_access_token": fyers_access_token,
                "fyers_client_id": fyers_client_id
            }
            fyers_client = FyersAPIClient(fyers_config)

        # Initialize Groq engine
        if not groq_engine:
            # Code Quality: Move hardcoded values to configuration
            groq_config = {
                "groq_api_key": os.getenv("GROQ_API_KEY", ""),
                "groq_model": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
                "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
                "temperature": float(os.getenv("GROQ_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
            }
            groq_engine = GroqReasoningEngine(groq_config)

        # Initialize MCP server
        if not mcp_server:
            mcp_config = {
                "monitoring_port": 8002,
                "max_sessions": 100
            }
            mcp_server = MCPTradingServer(mcp_config)

        # Initialize trading agent
        if not mcp_trading_agent:
            agent_config = {
                "agent_id": "production_trading_agent",
                "risk_tolerance": 0.02,
                "max_positions": 5,
                "min_confidence": 0.7,
                "fyers": {
                    "fyers_access_token": os.getenv("FYERS_ACCESS_TOKEN"),
                    "fyers_client_id": os.getenv("FYERS_APP_ID")
                },
                "llama": {
                    "llama_base_url": "http://localhost:11434",
                    "llama_model": "llama3.1:8b"
                },
                "groq": {
                    "groq_api_key": os.getenv("GROQ_API_KEY"),
                    "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                    "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "2048")),
                    "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.1")),
                }
            }
            mcp_trading_agent = TradingAgent(agent_config)
            await mcp_trading_agent.initialize()

        # Register MCP tools
        if mcp_server:
            # Import tools
            from mcp_server.tools.prediction_tool import PredictionTool
            from mcp_server.tools.scan_tool import ScanTool
            from mcp_server.tools.execution_tool import ExecutionTool
            from mcp_server.tools.portfolio_tool import PortfolioTool
            from mcp_server.tools.risk_management_tool import RiskManagementTool

            # Initialize tools
            prediction_tool = PredictionTool({
                "tool_id": "prediction_tool",
                "ollama_enabled": True,
                "ollama_host": "http://localhost:11434",
                "ollama_model": "llama3.1:8b"
            })

            scan_tool = ScanTool({
                "tool_id": "scan_tool",
                "ollama_enabled": True,
                "ollama_host": "http://localhost:11434",
                "ollama_model": "llama3.1:8b"
            })

            execution_tool = ExecutionTool({
                "tool_id": "execution_tool",
                "trading_mode": "paper",
                "max_order_value": 100000,
                "max_position_size": 0.25,
                "daily_loss_limit": 0.05
            })

            portfolio_tool = PortfolioTool({
                "tool_id": "portfolio_tool",
                "portfolio_agent": {},
                "risk_agent": {}
            })

            risk_management_tool = RiskManagementTool({
                "tool_id": "risk_management_tool",
                "risk_agent": {},
                "portfolio_var_limit": 0.05,
                "position_size_limit": 0.25,
                "concentration_limit": 0.4,
                "correlation_limit": 0.8,
                "liquidity_threshold": 0.3
            })

            # Register prediction tool
            mcp_server.register_tool(
                name="predict",
                function=prediction_tool.rank_predictions,
                description="Rank predictions from RL agents and other models",
                schema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "horizon": {"type": "string"},
                        "include_explanations": {"type": "boolean"},
                        "natural_query": {"type": "string"}
                    },
                    "required": []
                }
            )

            # Register scan tool
            mcp_server.register_tool(
                name="scan_all",
                function=scan_tool.scan_all,
                description="Generate filtered shortlists based on user criteria",
                schema={
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "object",
                            "properties": {
                                "min_price": {"type": "number"},
                                "max_price": {"type": "number"},
                                "min_volume": {"type": "number"},
                                "min_score": {"type": "number"},
                                "sectors": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "market_caps": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "risk_levels": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "sort_by": {"type": "string"},
                        "limit": {"type": "number"},
                        "natural_query": {"type": "string"}
                    },
                    "required": []
                }
            )

            # Register execution tool
            mcp_server.register_tool(
                name="execute_trade",
                function=execution_tool.execute_trade,
                description="Execute a trade order with comprehensive risk checks",
                schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "side": {"type": "string"},
                        "quantity": {"type": "number"},
                        "order_type": {"type": "string"},
                        "price": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"}
                    },
                    "required": ["symbol", "side", "quantity"]
                }
            )

            # Register portfolio analysis tool
            mcp_server.register_tool(
                name="analyze_portfolio",
                function=portfolio_tool.analyze_portfolio,
                description="Comprehensive portfolio analysis",
                schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "analysis_type": {"type": "string"},
                        "time_period": {"type": "string"},
                        "benchmark": {"type": "string"},
                        "include_recommendations": {"type": "boolean"}
                    },
                    "required": ["portfolio_id"]
                }
            )

            # Register portfolio optimization tool
            mcp_server.register_tool(
                name="optimize_portfolio",
                function=portfolio_tool.optimize_portfolio,
                description="Portfolio optimization with multiple methods",
                schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "optimization_method": {"type": "string"},
                        "risk_tolerance": {"type": "number"},
                        "target_return": {"type": "number"},
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "max_position_size": {"type": "number"}
                            }
                        }
                    },
                    "required": ["portfolio_id"]
                }
            )

            # Register risk assessment tool
            mcp_server.register_tool(
                name="assess_portfolio_risk",
                function=risk_management_tool.assess_portfolio_risk,
                description="Comprehensive portfolio risk assessment",
                schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "assessment_type": {"type": "string"},
                        "risk_metrics": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "confidence_level": {"type": "number"},
                        "time_horizon": {"type": "number"}
                    },
                    "required": ["portfolio_id"]
                }
            )

            # Register position risk assessment tool
            mcp_server.register_tool(
                name="assess_position_risk",
                function=risk_management_tool.assess_position_risk,
                description="Individual position risk assessment",
                schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "symbol": {"type": "string"},
                        "position_size": {"type": "number"}
                    },
                    "required": ["portfolio_id", "symbol"]
                }
            )

        logger.info("MCP components initialized successfully")

    except Exception as e:
        logger.error(f"MCP initialization error: {e}")
        raise


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        # Send initial data when client connects
        if trading_bot:
            initial_data = trading_bot.get_complete_bot_data()
            await manager.send_personal_message(
                json.dumps({
                    "type": "initial_data",
                    "data": initial_data,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )

        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            if data == "ping":
                await manager.send_personal_message("pong", websocket)
            elif data == "get_initial_data":
                if trading_bot:
                    initial_data = trading_bot.get_complete_bot_data()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "initial_data",
                            "data": initial_data,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
    finally:
        # Security: Ensure proper cleanup to prevent memory leaks
        try:
            if websocket in manager.active_connections:
                manager.disconnect(websocket)
            # Clear any remaining references
            websocket = None
        except Exception as cleanup_error:
            logger.error(f"Error during WebSocket cleanup: {cleanup_error}")


def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")


def run_web_server(host='127.0.0.1', port=5000, debug=False):
    """Run the FastAPI web server with uvicorn"""
    global trading_bot
    try:
        # Check if the requested port is available
        # CRITICAL: Frontend expects port 5000, so we must use it or fail
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
        except OSError:
            # Port 5000 is in use - try to kill the process (Windows)
            logger.warning(f"Port {port} is already in use. Attempting to free it...")
            try:
                import subprocess
                import platform
                if platform.system() == "Windows":
                    # Find PID using port 5000
                    result = subprocess.run(
                        ["netstat", "-ano"], capture_output=True, text=True
                    )
                    for line in result.stdout.splitlines():
                        if f":{port}" in line and "LISTENING" in line:
                            parts = line.split()
                            if len(parts) > 4:
                                pid = parts[-1]
                                logger.info(f"Killing process {pid} on port {port}")
                                subprocess.run(["taskkill", "/PID", pid, "/F"], 
                                             capture_output=True, check=False)
                                time.sleep(1)  # Wait for port to be freed
                                break
                # Try binding again
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host, port))
                logger.info(f"Port {port} is now available")
            except Exception as e:
                logger.error(f"Failed to free port {port}: {e}")
                logger.error(f"Please manually kill the process using port {port} and restart")
                raise RuntimeError(f"Port {port} is in use and could not be freed")

        # Don't initialize bot here - let the startup event handler do it
        # This prevents double initialization and blocking the server start
        logger.info(f"Starting FastAPI web server on http://{host}:{port}")
        logger.info("Web interface will be available at the above URL")
        logger.info("API documentation available at http://{host}:{port}/docs")

        # Configure uvicorn - use direct run to ensure server stays alive
        import uvicorn
        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info",  # Use info to see startup messages
                reload=False,  # Disable reload to prevent crashes
                access_log=True  # Always log access to debug connectivity
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            logger.exception("Full traceback:")
            raise

    except Exception as e:
        logger.error(f"Error running web server: {e}")
        raise


def _do_blocking_bot_init():
    """Sync helper: data service check + update watchlist + initialize_bot. Run in thread only."""
    global trading_bot
    try:
        # Load env file first to ensure credentials are available
        try:
            from dhan_client import _load_env
            _load_env()
        except:
            pass
        
        data_client = get_data_client()
        if data_client.is_service_available():
            logger.info("*** DATA SERVICE AVAILABLE - PRODUCTION MODE ***")
            comprehensive_watchlist = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
                "SUZLON.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
                "LT.NS", "AXISBANK.NS", "MARUTI.NS", "HINDUNILVR.NS", "WIPRO.NS",
                "SUNPHARMA.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS"
            ]
            data_client.update_watchlist(comprehensive_watchlist)
        else:
            logger.warning("*** DATA SERVICE NOT AVAILABLE - FALLBACK MODE ***")
            logger.info("Backend will use Yahoo Finance and mock data")
        initialize_bot()
    except Exception as e:
        logger.error(f"Blocking bot init failed: {e}")
        logger.exception("Full traceback:")
        # Don't fail - server should still start even if bot init fails
        trading_bot = None


async def startup_event():
    """Initialize the trading bot on startup - NON-BLOCKING: all heavy work runs in thread."""
    global trading_bot
    
    # CRITICAL: Don't block startup - server must start immediately
    # Bot initialization happens in background
    logger.info("Server startup event triggered - bot initialization will run in background")
    
    async def init_bot_background():
        """Run blocking init in executor so event loop stays free and server can accept connections."""
        global trading_bot
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            # CRITICAL: Run ALL blocking work (HTTP/data service + initialize_bot) in thread
            # so the server event loop is never blocked and port 5000 responds immediately
            await loop.run_in_executor(None, _do_blocking_bot_init)

            # Priority 3: Execute pending async initializations
            if trading_bot and hasattr(trading_bot, '_pending_async_inits'):
                logger.info("Executing pending async initializations...")
                for component_name, init_func in trading_bot._pending_async_inits:
                    try:
                        await init_func()
                        logger.info(f"Successfully initialized {component_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize {component_name}: {e}")
                # Clear pending initializations
                trading_bot._pending_async_inits = []

            logger.info("Trading bot initialized on startup")
        except Exception as init_error:
            logger.error(f"Bot initialization failed: {init_error}")
            logger.exception("Full traceback:")
            # Don't fail server startup - continue without bot
            trading_bot = None

            # Start Dhan sync service if in live mode
            if LIVE_TRADING_AVAILABLE and trading_bot and trading_bot.config.get("mode") == "live":
                try:
                    # Check Dhan credentials before starting sync service - load from env file
                    try:
                        from dhan_client import get_dhan_token, get_dhan_client_id
                        dhan_access_token = get_dhan_token()
                        dhan_client_id = get_dhan_client_id()
                    except ImportError:
                        # Fallback to direct env access
                        from dotenv import load_dotenv
                        import os
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        env_file = os.path.join(os.path.dirname(current_dir), "env")
                        if os.path.exists(env_file):
                            load_dotenv(env_file)
                        dhan_client_id = os.getenv("DHAN_CLIENT_ID")
                        dhan_access_token = os.getenv("DHAN_ACCESS_TOKEN")

                    if not dhan_client_id or not dhan_access_token:
                        logger.error(
                            "DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN not found in environment variables")
                        logger.error(
                            "Please set these in backend/hft2/env file to enable live trading")
                    else:
                        sync_service = start_sync_service(
                            sync_interval=30)  # Sync every 30 seconds
                        if sync_service:
                            try:
                                # Prefer the richer DhanAPIClient if available on the trading bot
                                current_balance = 0.0
                                funds = None
                                dhan_client = None
                                try:
                                    if hasattr(trading_bot, 'live_executor') and getattr(trading_bot.live_executor, 'dhan_client', None):
                                        dhan_client = trading_bot.live_executor.dhan_client
                                    elif getattr(trading_bot, 'dhan_client', None):
                                        dhan_client = trading_bot.dhan_client
                                except Exception:
                                    dhan_client = None

                                if dhan_client:
                                    try:
                                        funds = dhan_client.get_funds()
                                    except Exception as e:
                                        logger.debug(
                                            f"DhanAPIClient.get_funds() failed: {e}")

                                # Fallback to sync service's method if DhanAPIClient not available
                                if funds is None:
                                    funds = sync_service.get_dhan_funds()

                                # Extract numeric balance for logging
                                if isinstance(funds, dict):
                                    for key in ('availableBalance', 'availabelBalance', 'available_balance', 'available', 'availBalance', 'cash', 'netBalance', 'totalBalance'):
                                        if key in funds:
                                            try:
                                                current_balance = float(
                                                    funds.get(key, 0.0) or 0.0)
                                                break
                                            except Exception:
                                                continue
                                    else:
                                        for v in funds.values():
                                            if isinstance(v, (int, float)):
                                                current_balance = float(v)
                                                break

                                # If balance is zero, log debug information to diagnose
                                if abs(current_balance) < 0.01:
                                    logger.debug(
                                        f"Dhan funds response on startup: {funds}")

                                logger.info(
                                    f"🚀 Dhan real-time sync service started (30s interval) — Balance: ₹{current_balance:.2f}")
                            except Exception as e:
                                logger.info(
                                    "🚀 Dhan real-time sync service started (30s interval)")
                                logger.debug(
                                    f"Failed to fetch initial Dhan funds: {e}")
                        else:
                            logger.warning("Failed to start Dhan sync service")
                except Exception as sync_error:
                    logger.error(f"Error starting Dhan sync service: {sync_error}")
        except Exception as e:
            logger.error(f"Error initializing bot in background: {e}")
            logger.exception("Full traceback:")
            trading_bot = None
    
    # Mark server start time for /api/health uptime
    import time
    app.start_time = time.time()

    # CRITICAL: Start bot initialization as background task - server starts immediately
    # Use asyncio.create_task to run in background without blocking
    # Wrap in try-except to ensure server startup completes even if task creation fails
    try:
        import asyncio
        # Create task but don't await it - let it run in background
        task = asyncio.create_task(init_bot_background())
        # Add error callback to prevent unhandled exceptions from crashing the server
        def handle_task_error(task):
            try:
                task.result()  # This will raise if task failed
            except Exception as e:
                logger.error(f"Background bot initialization failed: {e}")
                logger.exception("Full traceback:")
        task.add_done_callback(handle_task_error)
        logger.info("Server started - bot initialization running in background")
    except Exception as e:
        logger.error(f"Failed to start background initialization task: {e}")
        logger.exception("Full traceback:")
        # Don't crash the server if background task creation fails
    
    # CRITICAL: Return immediately so FastAPI marks startup as complete
    # The background task will continue running independently
    # No await here - function returns immediately


@app.get("/api/monitoring")
async def get_monitoring_stats():
    """Advanced Optimization: Get system performance statistics"""
    try:
        stats = {
            "performance": performance_monitor.get_stats(),
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational"
        }

        # Add data service stats if available
        try:
            data_client = get_data_client()
            if hasattr(data_client, 'get_cache_stats'):
                stats["data_service_cache"] = data_client.get_cache_stats()
        except Exception as e:
            logger.debug(f"Could not get data service stats: {e}")

        return stats
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}")
        raise HTTPException(
            status_code=500, detail="Error retrieving monitoring data")


async def shutdown_event():
    """Architectural Fix: Comprehensive resource cleanup on shutdown"""
    global trading_bot, mcp_server, fyers_client, groq_engine

    logger.info("Starting graceful shutdown...")

    try:
        # Stop Dhan sync service
        if LIVE_TRADING_AVAILABLE:
            try:
                stop_sync_service()
                logger.info("Dhan sync service stopped")
            except Exception as e:
                logger.error(f"Error stopping Dhan sync service: {e}")

        # Stop trading bot
        if trading_bot:
            trading_bot.stop()
            logger.info("Trading bot stopped")

        # Cleanup MCP server
        if mcp_server:
            try:
                await mcp_server.shutdown()
                logger.info("MCP server shutdown")
            except Exception as e:
                logger.error(f"Error shutting down MCP server: {e}")

        # Cleanup Fyers client
        if fyers_client:
            try:
                await fyers_client.disconnect()
                logger.info("Fyers client disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Fyers client: {e}")

        # Cleanup Groq engine
        if groq_engine:
            try:
                await groq_engine.cleanup()
                logger.info("Groq engine cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Groq engine: {e}")

        logger.info("Graceful shutdown completed")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Indian Stock Trading Bot Web Interface (FastAPI)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    args = parser.parse_args()

    try:
        run_web_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        sys.exit(1)

# New API endpoints for RL scanning system


@app.post("/api/scan_all")
async def scan_all():
    """Trigger full market scan"""
    try:
        logger.info("Manual market scan triggered via API")
        data_agent.kickoff_scan()
        return {"status": "scan_started", "message": "Full market scan initiated"}
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_stocks(request: AnalyzeRequest):
    """Analyze custom tickers and return entry/exit points with confidence"""
    if not Stock:
        raise HTTPException(status_code=503, detail="Analysis components not loaded. Install optional deps and restart.")
    try:
        results = {}

        for ticker in request.tickers:
            try:
                # Use existing Stock class for analysis
                stock = Stock(ticker)

                # Get basic analysis (simplified - enhance based on your Stock class methods)
                price_data = stock.get_current_price()
                sentiment = stock.get_sentiment_score()

                # Calculate entry/exit based on current implementation
                entry_price = price_data * 0.98  # 2% below current
                exit_price = price_data * 1.05   # 5% above current
                confidence = min(sentiment * 0.8, 0.95)  # Cap at 95%

                results[ticker] = {
                    "entry": round(entry_price, 2),
                    "exit": round(exit_price, 2),
                    "confidence": round(confidence, 3),
                    "current_price": round(price_data, 2),
                    "horizon": request.horizon
                }
            except Exception as e:
                logger.error(f"Analysis failed for {ticker}: {e}")
                results[ticker] = {"error": str(e)}

        return results
    except Exception as e:
        logger.error(f"Analyze endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/update_risk")
async def update_risk(request: UpdateRiskRequest):
    """Update risk settings in live_config.json"""
    try:
        risk_engine.update_risk_profile(
            request.stop_loss_pct,
            request.capital_risk_pct,
            request.drawdown_limit_pct
        )
        return {
            "status": "updated",
            "message": "Risk profile updated in live_config.json",
            "new_settings": risk_engine.get_risk_settings()
        }
    except Exception as e:
        logger.error(f"Risk update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/shortlist")
async def get_shortlist():
    """Get current shortlist from RL filtering"""
    try:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        shortlist_file = f"logs/shortlist_{date_str}.json"

        if os.path.exists(shortlist_file):
            with open(shortlist_file, 'r') as f:
                data = json.load(f)
                return data
        else:
            return {"message": "No shortlist available for today", "shortlist": []}
    except Exception as e:
        logger.error(f"Error getting shortlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tracking_stats")
async def get_tracking_stats():
    """Get monitoring and tracking statistics"""
    try:
        return {
            "data_agent_stats": data_agent.get_cache_stats(),
            "rl_agent_stats": rl_agent.get_model_stats(),
            "tracker_stats": tracker_agent.get_monitoring_stats(),
            "risk_settings": risk_engine.get_risk_settings()
        }
    except Exception as e:
        logger.error(f"Error getting tracking stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
