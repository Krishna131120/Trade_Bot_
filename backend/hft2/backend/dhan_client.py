"""
Load env from backend/hft2/env and fetch live portfolio from Dhan API.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Load env file from ../env (parent of backend/)
_ENV_LOADED = False


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    for env_path in [Path(__file__).resolve().parent.parent / "env", Path(__file__).resolve().parent / ".env"]:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "=" in line and not line.startswith("#"):
                            k, _, v = line.partition("=")
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            os.environ[k] = v
                _ENV_LOADED = True
                logger.info("Loaded env from %s", env_path)
                break
            except Exception as e:
                logger.warning("Failed to load %s: %s", env_path, e)
    _ENV_LOADED = True


def get_dhan_token() -> Optional[str]:
    _load_env()
    return os.environ.get("DHAN_ACCESS_TOKEN") or os.environ.get("DHAN_ACCESS_TOKEN_KEY")


def get_dhan_client_id() -> Optional[str]:
    _load_env()
    return os.environ.get("DHAN_CLIENT_ID")


def _dhan_request(method: str, path: str, token: str, **kwargs: Any) -> Any:
    import urllib.request
    import json
    url = f"https://api.dhan.co/v2{path}"
    req = urllib.request.Request(url, method=method, headers={
        "Content-Type": "application/json",
        "access-token": token,
    })
    if kwargs.get("data"):
        req.data = json.dumps(kwargs["data"]).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        logger.warning("Dhan API %s %s failed: %s", method, path, e)
        return None


def fetch_fund_limit(token: str) -> Optional[Dict]:
    out = _dhan_request("GET", "/fundlimit", token)
    return out if isinstance(out, dict) else None


def fetch_holdings(token: str) -> List[Dict]:
    out = _dhan_request("GET", "/holdings", token)
    if isinstance(out, list):
        return out
    return []


def fetch_positions(token: str) -> List[Dict]:
    out = _dhan_request("GET", "/positions", token)
    if isinstance(out, list):
        return out
    return []


def _nse_symbol(s: str, segment: str = "") -> str:
    if not s:
        return s
    if "NSE" in segment or segment == "NSE_EQ" or not segment:
        return f"{s}.NS" if "." not in s else s
    if "BSE" in segment:
        return f"{s}.BO" if "." not in s else s
    return s


def _get_fyers_ltp(symbol: str) -> Optional[float]:
    """Fetch LTP for symbol from Fyers data service (port 8002). Returns price or None."""
    _load_env()
    import urllib.request
    import json
    from urllib.parse import quote
    port = os.environ.get("DATA_SERVICE_PORT", "8002")
    base = f"http://127.0.0.1:{port}"
    url = f"{base}/data/{quote(symbol, safe='')}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as r:
            data = json.loads(r.read().decode())
            if isinstance(data, dict) and "price" in data:
                p = float(data["price"])
                return p if p > 0 else None
    except Exception as e:
        logger.debug("Fyers LTP for %s: %s", symbol, e)
    return None


def get_live_portfolio() -> Optional[Dict]:
    """Build portfolio dict for HFT dashboard from Dhan: holdings, cash, totalValue, tradeLog."""
    token = get_dhan_token()
    client_id = get_dhan_client_id()
    logger.info(f"[get_live_portfolio] Token: {bool(token)}, Client ID: {bool(client_id)}")
    if not token:
        logger.warning("[get_live_portfolio] No Dhan token - cannot fetch portfolio")
        return None
    try:
        fund = fetch_fund_limit(token)
        holdings_list = fetch_holdings(token)
        positions_list = fetch_positions(token)
        logger.info(f"[get_live_portfolio] Dhan API: fund={bool(fund)}, holdings={len(holdings_list)}, positions={len(positions_list)}")
    except Exception as e:
        logger.error(f"[get_live_portfolio] Dhan API error: {e}")
        return None
    cash = 0.0
    if fund and "availabelBalance" in fund:
        cash = float(fund["availabelBalance"])
    elif fund and "availableBalance" in fund:
        cash = float(fund["availableBalance"])
    sod = float(fund.get("sodLimit", 0) or 0) if fund else 0

    holdings: Dict[str, Dict] = {}
    # Delivery holdings (demat)
    for h in holdings_list:
        sym = _nse_symbol(str(h.get("tradingSymbol", "")), str(h.get("exchange", "NSE_EQ")))
        qty = int(h.get("availableQty") or h.get("totalQty") or 0)
        if qty <= 0 or not sym:
            continue
        avg = float(h.get("avgCostPrice") or 0)
        holdings[sym] = {
            "symbol": sym,
            "quantity": qty,
            "avgPrice": avg,
            "currentPrice": avg,
            "lastAction": "BUY",
        }
    # Open positions (intraday / today)
    for p in positions_list:
        if str(p.get("positionType")) == "CLOSED":
            continue
        sym = _nse_symbol(str(p.get("tradingSymbol", "")), str(p.get("exchangeSegment", "NSE_EQ")))
        net_qty = int(p.get("netQty") or 0)
        if net_qty == 0 or not sym:
            continue
        buy_avg = float(p.get("buyAvg") or 0)
        cost_price = float(p.get("costPrice") or buy_avg)
        unrealized = float(p.get("unrealizedProfit") or 0)
        # currentPrice such that (currentPrice - costPrice) * netQty = unrealizedProfit
        if net_qty and cost_price:
            current_price = cost_price + (unrealized / net_qty)
        else:
            current_price = buy_avg
        if sym in holdings:
            # Merge with existing holding (e.g. same script in CNC + intraday)
            old = holdings[sym]
            tot_qty = old["quantity"] + net_qty
            old_avg = old["avgPrice"] * old["quantity"] + cost_price * net_qty
            holdings[sym]["quantity"] = tot_qty
            holdings[sym]["avgPrice"] = old_avg / tot_qty if tot_qty else old["avgPrice"]
            holdings[sym]["currentPrice"] = current_price
        else:
            holdings[sym] = {
                "symbol": sym,
                "quantity": net_qty,
                "avgPrice": cost_price,
                "currentPrice": current_price,
                "lastAction": "BUY" if net_qty > 0 else "SELL",
            }
    # Live LTP from Fyers data service for every symbol (P&L and portfolio value real-time)
    for sym in list(holdings.keys()):
        ltp = _get_fyers_ltp(sym)
        if ltp is not None and ltp > 0:
            holdings[sym]["currentPrice"] = round(ltp, 2)

    equity_value = sum((h["currentPrice"] or h["avgPrice"]) * h["quantity"] for h in holdings.values())
    total_value = cash + equity_value
    starting_balance = sod if sod > 0 else total_value

    result = {
        "totalValue": round(total_value, 2),
        "cash": round(cash, 2),
        "startingBalance": round(starting_balance, 2),
        "holdings": holdings,
        "tradeLog": [],
    }
    logger.info(f"[get_live_portfolio] Final portfolio: totalValue={result['totalValue']}, holdings={len(holdings)}, symbols={list(holdings.keys())}")
    return result


def get_live_trades(limit: int = 50) -> List[Dict]:
    """Dhan doesn't expose trade history in same API; return empty or derive from positions."""
    return []


# ---------------------------------------------------------------------------
# DhanAPIClient class for web_backend.py / live_executor.py (same env, same API)
# ---------------------------------------------------------------------------

class DhanAPIClient:
    """Dhan API client used by live_executor and web_backend. Uses backend/hft2/env for credentials."""

    def __init__(self, client_id: str, access_token: str):
        _load_env()
        self.client_id = client_id or get_dhan_client_id()
        self.access_token = access_token or get_dhan_token()
        if not self.access_token:
            raise ValueError("DHAN_ACCESS_TOKEN required")

    def validate_connection(self) -> bool:
        fund = fetch_fund_limit(self.access_token)
        return fund is not None and isinstance(fund, dict)

    def get_funds(self) -> Dict[str, Any]:
        """Return fund limit dict (availabelBalance, sodLimit, etc.)."""
        fund = fetch_fund_limit(self.access_token)
        if fund is not None and isinstance(fund, dict):
            return fund
        return {}

    def get_holdings(self) -> List[Dict]:
        """Return raw Dhan holdings list (tradingSymbol, totalQty, avgCostPrice, etc.)."""
        return fetch_holdings(self.access_token)

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """LTP/quote for symbol. Not implemented here; returns None."""
        return None

    def place_order(self, **kwargs: Any) -> Any:
        """Place order via Dhan. Stub; implement with Dhan order API if needed."""
        logger.warning("DhanAPIClient.place_order not implemented")
        return None

    def get_orders(self) -> List[Dict]:
        """Fetch orders. Stub; implement with Dhan orders API if needed."""
        return []

    def get_market_status(self) -> Optional[Dict]:
        """Market status. Stub."""
        return None

    def get_profile(self) -> Optional[Dict]:
        """User profile. Stub."""
        return None

    def is_market_open(self) -> bool:
        """Stub; assume open during market hours if needed."""
        return True


def place_order_market(symbol: str, side: str, quantity: int, product_type: str = "CNC", trigger_price: Optional[float] = None) -> Optional[Dict]:
    """Place MARKET order via Dhan. Resolves securityId from holdings. Returns order response or None."""
    token = get_dhan_token()
    client_id = get_dhan_client_id()
    if not token or not client_id:
        return None
    holdings_list = fetch_holdings(token)
    # Symbol from frontend e.g. RELIANCE.NS or TCS.NS; Dhan uses RELIANCE, TCS
    sym_clean = (symbol or "").replace(".NS", "").replace(".BO", "").strip().upper()
    security_id = None
    exchange_segment = "NSE_EQ"
    for h in holdings_list:
        if str(h.get("tradingSymbol", "")).upper() == sym_clean:
            security_id = str(h.get("securityId", ""))
            break
    if not security_id:
        # Try positions for intraday symbols
        for p in fetch_positions(token):
            if str(p.get("tradingSymbol", "")).upper() == sym_clean:
                security_id = str(p.get("securityId", ""))
                exchange_segment = str(p.get("exchangeSegment", "NSE_EQ"))
                break
    if not security_id:
        logger.warning("Could not resolve securityId for %s (not in holdings/positions)", symbol)
        return None
    body = {
        "dhanClientId": client_id,
        "transactionType": side.upper(),
        "exchangeSegment": exchange_segment,
        "productType": product_type,
        "orderType": "MARKET",
        "validity": "DAY",
        "securityId": security_id,
        "quantity": quantity,
        "price": 0,
        "disclosedQuantity": "",
        "triggerPrice": trigger_price or "",
        "afterMarketOrder": False,
        "amoTime": "",
        "boProfitValue": "",
        "boStopLossValue": "",
    }
    out = _dhan_request("POST", "/orders", token, data=body)
    return out if isinstance(out, dict) else None
