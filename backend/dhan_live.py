"""
Minimal Dhan API client for the main backend (Render / single-service deploy).
Uses only os.environ - no env file. Set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID in Render env.
No Fyers dependency; currentPrice comes from Dhan position data (unrealized P&L).
"""
import os
import logging
import urllib.request
import json
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_dhan_token() -> Optional[str]:
    return os.environ.get("DHAN_ACCESS_TOKEN") or os.environ.get("DHAN_ACCESS_TOKEN_KEY")


def get_dhan_client_id() -> Optional[str]:
    return os.environ.get("DHAN_CLIENT_ID")


def _dhan_request(method: str, path: str, token: str, **kwargs: Any) -> Any:
    url = f"https://api.dhan.co/v2{path}"
    req = urllib.request.Request(
        url,
        method=method,
        headers={"Content-Type": "application/json", "access-token": token},
    )
    if kwargs.get("data"):
        req.data = json.dumps(kwargs["data"]).encode("utf-8")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        logger.warning("Dhan API %s %s failed: %s", method, path, e)
        return None


def _fetch_fund_limit(token: str) -> Optional[Dict]:
    out = _dhan_request("GET", "/fundlimit", token)
    return out if isinstance(out, dict) else None


def _fetch_holdings(token: str) -> List[Dict]:
    out = _dhan_request("GET", "/holdings", token)
    return out if isinstance(out, list) else []


def _fetch_positions(token: str) -> List[Dict]:
    out = _dhan_request("GET", "/positions", token)
    return out if isinstance(out, list) else []


def _nse_symbol(s: str, segment: str = "") -> str:
    if not s:
        return s
    if "NSE" in segment or segment == "NSE_EQ" or not segment:
        return f"{s}.NS" if "." not in s else s
    if "BSE" in segment:
        return f"{s}.BO" if "." not in s else s
    return s


def get_live_portfolio() -> Optional[Dict]:
    """Build portfolio dict for HFT dashboard from Dhan. Uses env only (Render-safe)."""
    token = get_dhan_token()
    if not token:
        return None
    try:
        fund = _fetch_fund_limit(token)
        holdings_list = _fetch_holdings(token)
        positions_list = _fetch_positions(token)
    except Exception as e:
        logger.error("Dhan API error: %s", e)
        return None

    cash = 0.0
    if fund and "availabelBalance" in fund:
        cash = float(fund["availabelBalance"])
    elif fund and "availableBalance" in fund:
        cash = float(fund["availableBalance"])
    sod = float(fund.get("sodLimit", 0) or 0) if fund else 0

    holdings: Dict[str, Dict] = {}
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
        current_price = cost_price + (unrealized / net_qty) if (net_qty and cost_price) else buy_avg
        if sym in holdings:
            old = holdings[sym]
            tot_qty = old["quantity"] + net_qty
            old_avg = old["avgPrice"] * old["quantity"] + cost_price * net_qty
            holdings[sym]["quantity"] = tot_qty
            holdings[sym]["avgPrice"] = old_avg / tot_qty if tot_qty else old["avgPrice"]
            holdings[sym]["currentPrice"] = round(current_price, 2)
        else:
            holdings[sym] = {
                "symbol": sym,
                "quantity": net_qty,
                "avgPrice": cost_price,
                "currentPrice": round(current_price, 2),
                "lastAction": "BUY" if net_qty > 0 else "SELL",
            }

    equity_value = sum((h.get("currentPrice") or h["avgPrice"]) * h["quantity"] for h in holdings.values())
    total_value = cash + equity_value
    starting_balance = sod if sod > 0 else total_value

    return {
        "totalValue": round(total_value, 2),
        "cash": round(cash, 2),
        "startingBalance": round(starting_balance, 2),
        "holdings": holdings,
        "tradeLog": [],
    }
