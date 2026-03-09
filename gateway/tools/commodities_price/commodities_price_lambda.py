import hashlib
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = Path("/tmp/av_cache")  # noqa: S108  # nosec B108
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTLs in seconds
CACHE_TTL_SPOT = 300  # 5 min for live spot prices
CACHE_TTL_HISTORICAL = 21600  # 6 hours for historical data
CACHE_TTL_ECONOMIC = 21600  # 6 hours for economic indicators
CACHE_TTL_NEWS = 1800  # 30 min for news sentiment

# Rate limiting — persists across invocations via /tmp
RATE_LIMIT_FILE = CACHE_DIR / "_last_api_call.txt"
MIN_API_INTERVAL = 1.5  # seconds between API calls

# Commodities: function=<NAME>&interval=<interval>
COMMODITY_FUNCTIONS = {
    "WTI": {"function": "WTI", "name": "WTI Crude Oil"},
    "BRENT": {"function": "BRENT", "name": "Brent Crude Oil"},
    "NATURAL_GAS": {"function": "NATURAL_GAS", "name": "Natural Gas"},
    "COPPER": {"function": "COPPER", "name": "Copper"},
    "ALUMINUM": {"function": "ALUMINUM", "name": "Aluminum"},
    "WHEAT": {"function": "WHEAT", "name": "Wheat"},
    "CORN": {"function": "CORN", "name": "Corn"},
    "COTTON": {"function": "COTTON", "name": "Cotton"},
    "SUGAR": {"function": "SUGAR", "name": "Sugar"},
    "COFFEE": {"function": "COFFEE", "name": "Coffee"},
}

# Precious metals use separate endpoints
PRECIOUS_METALS = {
    "GOLD": {"symbol": "GOLD", "name": "Gold (XAU/USD)"},
    "SILVER": {"symbol": "SILVER", "name": "Silver (XAG/USD)"},
}

ALL_COMMODITIES = {
    **{k: v["name"] for k, v in COMMODITY_FUNCTIONS.items()},
    **{k: v["name"] for k, v in PRECIOUS_METALS.items()},
}

# Economic indicators
ECONOMIC_INDICATORS = {
    "CPI": {
        "function": "CPI",
        "name": "Consumer Price Index",
        "default_interval": "monthly",
    },
    "INFLATION": {
        "function": "INFLATION",
        "name": "Inflation Rate",
        "default_interval": "annual",
    },
    "FEDERAL_FUNDS_RATE": {
        "function": "FEDERAL_FUNDS_RATE",
        "name": "Federal Funds Rate",
        "default_interval": "monthly",
    },
    "TREASURY_YIELD": {
        "function": "TREASURY_YIELD",
        "name": "Treasury Yield",
        "default_interval": "monthly",
    },
    "REAL_GDP": {
        "function": "REAL_GDP",
        "name": "Real GDP",
        "default_interval": "quarterly",
    },
    "REAL_GDP_PER_CAPITA": {
        "function": "REAL_GDP_PER_CAPITA",
        "name": "Real GDP Per Capita",
        "default_interval": "quarterly",
    },
    "UNEMPLOYMENT": {
        "function": "UNEMPLOYMENT",
        "name": "Unemployment Rate",
        "default_interval": "monthly",
    },
    "RETAIL_SALES": {
        "function": "RETAIL_SALES",
        "name": "Retail Sales",
        "default_interval": "monthly",
    },
    "NONFARM_PAYROLL": {
        "function": "NONFARM_PAYROLL",
        "name": "Nonfarm Payroll",
        "default_interval": "monthly",
    },
    "DURABLE_GOODS": {
        "function": "DURABLE_GOODS",
        "name": "Durable Goods Orders",
        "default_interval": "monthly",
    },
}


# ============================================================
# Cache + rate limiting
# ============================================================


def _cache_key(params: dict) -> str:
    cache_params = {k: v for k, v in params.items() if k != "apikey"}
    raw = json.dumps(cache_params, sort_keys=True)
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()  # noqa: S324


def _cache_get(key: str, ttl: int) -> dict | None:
    cache_file = CACHE_DIR / f"{key}.json"
    try:
        data = json.loads(cache_file.read_text())
        if time.time() - data.get("_cached_at", 0) < ttl:
            logger.info(f"Cache HIT for {key}")
            return data.get("payload")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return None


def _cache_set(key: str, payload: dict) -> None:
    try:
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps({"_cached_at": time.time(), "payload": payload})
        )
    except OSError as e:
        logger.warning(f"Cache write failed: {e}")


def _get_last_api_time() -> float:
    try:
        return float(RATE_LIMIT_FILE.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        pass
    return 0.0


def _set_last_api_time() -> None:
    try:
        RATE_LIMIT_FILE.write_text(str(time.time()))
    except OSError:
        pass


def get_api_error(data: dict) -> str | None:
    for key in ("Error Message", "Note", "Information"):
        if key in data:
            return data[key]
    return None


def fetch_cached(params: dict, ttl: int) -> dict:
    """Fetch from Alpha Vantage with caching and rate limiting."""
    key = _cache_key(params)

    cached = _cache_get(key, ttl)
    if cached is not None:
        return cached

    # Enforce delay between API calls
    last_call = _get_last_api_time()
    elapsed = time.time() - last_call
    if elapsed < MIN_API_INTERVAL:
        wait = MIN_API_INTERVAL - elapsed
        logger.info(f"Rate limiting: waiting {wait:.1f}s")
        time.sleep(wait)

    url = f"{ALPHA_VANTAGE_URL}?{urlencode(params)}"
    req = Request(url, headers={"Accept": "application/json"}, method="GET")  # noqa: S310

    with urlopen(req, timeout=30) as resp:  # noqa: S310  # nosec B310
        data = json.loads(resp.read().decode("utf-8"))
    _set_last_api_time()

    logger.info(f"API {params.get('function', '?')}: keys={list(data.keys())}")

    if not get_api_error(data):
        _cache_set(key, data)

    return data


_sm_client = boto3.client(
    "secretsmanager",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
)
_cached_api_key: str | None = None


def get_api_key() -> str:
    global _cached_api_key
    if _cached_api_key is None:
        secret_name = os.environ.get("COMMODITIES_SECRET_NAME", "commodities-api-key")
        response = _sm_client.get_secret_value(SecretId=secret_name)
        _cached_api_key = response["SecretString"]
    return _cached_api_key


# ============================================================
# Data fetching functions
# ============================================================


def fetch_spot_prices(symbols: list[str], api_key: str) -> str:
    """Fetch latest prices for commodities."""
    output = "## Latest Commodity Prices\n\n"

    for symbol in symbols:
        s = symbol.upper()

        if s in PRECIOUS_METALS:
            metal = PRECIOUS_METALS[s]
            data = fetch_cached(
                {
                    "function": "GOLD_SILVER_SPOT",
                    "symbol": metal["symbol"],
                    "apikey": api_key,
                },
                CACHE_TTL_SPOT,
            )

            error = get_api_error(data)
            if error:
                output += f"- **{metal['name']}:** {error}\n"
            elif data.get("nominal") == "invalid" or not data.get("price"):
                output += f"- **{metal['name']}:** No data available\n"
            else:
                price = float(data["price"])
                ts = data.get("timestamp", "unknown")
                output += (
                    f"- **{metal['name']}:** ${price:,.2f}"
                    f" per troy ounce — as of {ts}\n"
                )

        elif s in COMMODITY_FUNCTIONS:
            commodity = COMMODITY_FUNCTIONS[s]
            data = fetch_cached(
                {
                    "function": commodity["function"],
                    "interval": "monthly",
                    "apikey": api_key,
                },
                CACHE_TTL_HISTORICAL,
            )

            error = get_api_error(data)
            if error:
                output += f"- **{commodity['name']}:** {error}\n"
            else:
                points = data.get("data", [])
                latest = points[0] if points else {}
                name = data.get("name", commodity["name"])
                unit = data.get("unit", "")
                value = latest.get("value", ".")
                if value and value != ".":
                    date = latest.get("date", "?")
                    output += (
                        f"- **{name}:** ${float(value):,.2f} ({unit}) — as of {date}\n"
                    )
                else:
                    output += f"- **{name}:** No price available\n"
        else:
            supported = ", ".join(sorted(ALL_COMMODITIES))
            output += f"- **{symbol}:** Unknown. Supported: {supported}\n"

    return output + "\n"


def fetch_historical(
    symbols: list[str], start_date: str, end_date: str, api_key: str
) -> str:
    """Fetch historical prices for a date range."""
    output = f"## Historical Commodity Prices ({start_date} to {end_date})\n\n"

    for symbol in symbols:
        s = symbol.upper()

        if s in PRECIOUS_METALS:
            metal = PRECIOUS_METALS[s]
            data = fetch_cached(
                {
                    "function": "GOLD_SILVER_HISTORY",
                    "symbol": metal["symbol"],
                    "interval": "monthly",
                    "apikey": api_key,
                },
                CACHE_TTL_HISTORICAL,
            )

            error = get_api_error(data)
            if error:
                output += f"### {metal['name']}\n{error}\n\n"
                continue
            if data.get("nominal") == "invalid":
                output += f"### {metal['name']}\nNo data available\n\n"
                continue

            filtered = [
                p
                for p in data.get("data", [])
                if start_date <= p.get("date", "") <= end_date and p.get("price")
            ]
            if not filtered:
                output += f"### {metal['name']}\nNo data for this date range\n\n"
                continue

            output += f"### {metal['name']} (USD per troy ounce)\n"
            output += _format_table(filtered, "price", "dollars")

        elif s in COMMODITY_FUNCTIONS:
            commodity = COMMODITY_FUNCTIONS[s]
            data = fetch_cached(
                {
                    "function": commodity["function"],
                    "interval": "monthly",
                    "apikey": api_key,
                },
                CACHE_TTL_HISTORICAL,
            )

            error = get_api_error(data)
            if error:
                output += f"### {commodity['name']}\n{error}\n\n"
                continue

            name = data.get("name", commodity["name"])
            unit = data.get("unit", "")
            filtered = [
                p
                for p in data.get("data", [])
                if start_date <= p.get("date", "") <= end_date
                and p.get("value", ".") != "."
            ]
            if not filtered:
                output += f"### {name}\nNo data for this date range\n\n"
                continue

            output += f"### {name} ({unit})\n"
            output += _format_table(filtered, "value", unit)
        else:
            supported = ", ".join(sorted(ALL_COMMODITIES))
            output += f"### {symbol}\nUnknown. Supported: {supported}\n\n"

    return output


def fetch_indicators(
    indicators: list[str], start_date: str, end_date: str, api_key: str
) -> str:
    """Fetch economic indicators."""
    output = "## Economic Indicators"
    if start_date and end_date:
        output += f" ({start_date} to {end_date})"
    output += "\n\n"

    for indicator in indicators:
        config = ECONOMIC_INDICATORS.get(indicator.upper())
        if not config:
            supported = ", ".join(sorted(ECONOMIC_INDICATORS))
            output += f"### {indicator}\nUnknown. Supported: {supported}\n\n"
            continue

        params: dict = {"function": config["function"], "apikey": api_key}
        if config["function"] != "INFLATION":
            params["interval"] = config["default_interval"]
        if config["function"] == "TREASURY_YIELD":
            params["maturity"] = "10year"

        data = fetch_cached(params, CACHE_TTL_ECONOMIC)

        error = get_api_error(data)
        if error:
            output += f"### {config['name']}\n{error}\n\n"
            continue

        points = data.get("data", [])
        name = data.get("name", config["name"])
        unit = data.get("unit", "")

        if start_date and end_date:
            filtered = [
                p
                for p in points
                if start_date <= p.get("date", "") <= end_date
                and p.get("value", ".") != "."
            ]
        else:
            filtered = [p for p in points if p.get("value", ".") != "."][:12]

        if not filtered:
            output += f"### {name}\nNo data for this period\n\n"
            continue

        output += f"### {name} ({unit})\n"
        output += _format_table(filtered, "value", unit)

    return output


def fetch_news(topics: list[str], limit: int, api_key: str) -> str:
    """Fetch market news with sentiment."""
    params: dict = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "limit": min(limit, 50),
        "sort": "RELEVANCE",
    }
    if topics:
        params["topics"] = ",".join(topics)

    data = fetch_cached(params, CACHE_TTL_NEWS)

    error = get_api_error(data)
    if error:
        return f"## Market News Sentiment\n\n{error}\n\n"

    feed = data.get("feed", [])
    if not feed:
        return "## Market News Sentiment\n\nNo news articles found.\n\n"

    output = "## Market News Sentiment\n\n"
    for article in feed:
        title = article.get("title", "No title")
        url = article.get("url", "")
        source = article.get("source", "Unknown")
        published = article.get("time_published", "")
        summary = article.get("summary", "")[:300]
        score = article.get("overall_sentiment_score", 0)
        label = article.get("overall_sentiment_label", "Neutral")

        date_str = (
            f"{published[:4]}-{published[4:6]}-{published[6:8]}"
            if len(published) >= 8
            else published
        )

        output += f"### {title}\n"
        output += (
            f"**Source:** {source} | **Date:** {date_str}"
            f" | **Sentiment:** {label} ({score:+.3f})\n\n"
        )
        output += f"{summary}...\n\n[Source: {url}]\n\n"

    return output


def _format_table(points: list[dict], value_key: str, unit: str) -> str:
    """Format data as markdown table with summary stats."""
    is_dollar = unit and "percent" not in unit.lower() and "index" not in unit.lower()
    prefix = "$" if is_dollar else ""

    output = "| Date | Value |\n|------|-------|\n"
    values = []
    for p in sorted(points, key=lambda x: x["date"]):
        val = float(p[value_key])
        values.append(val)
        output += f"| {p['date']} | {prefix}{val:,.2f} |\n"

    if values:
        output += f"\n**Min:** {prefix}{min(values):,.2f} | "
        output += f"**Max:** {prefix}{max(values):,.2f} | "
        output += f"**Avg:** {prefix}{sum(values) / len(values):,.2f} | "
        output += f"**Data points:** {len(values)}\n\n"

    return output


# ============================================================
# Handler — single unified tool
# ============================================================


def _ensure_list(val) -> list[str]:
    """Coerce a string or list to a list of trimmed strings."""
    if isinstance(val, str):
        return [s.strip() for s in val.split(",")]
    return val if val else []


def handler(event, context):
    """
    AlphaVantage unified research tool for AgentCore Gateway.

    Fetches commodity prices, economic indicators, and market news
    in a single Lambda invocation with proper rate limiting.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        delimiter = "___"
        original_tool_name = context.client_context.custom["bedrockAgentCoreToolName"]
        tool_name = original_tool_name[
            original_tool_name.index(delimiter) + len(delimiter) :
        ]
        logger.info(f"Processing tool: {tool_name}")

        if tool_name != "alphavantage_research":
            return {
                "error": f"Unknown tool: {tool_name}. Expected: alphavantage_research"
            }

        api_key = get_api_key()
        output = ""

        # 1. Spot prices
        spot_symbols = _ensure_list(event.get("spot_symbols", []))
        if spot_symbols:
            output += fetch_spot_prices(spot_symbols, api_key)

        # 2. Historical prices
        historical_symbols = _ensure_list(event.get("historical_symbols", []))
        start_date = event.get("start_date", "")
        end_date = event.get("end_date", "")
        if historical_symbols and start_date and end_date:
            output += fetch_historical(
                historical_symbols, start_date, end_date, api_key
            )

        # 3. Economic indicators
        indicators = _ensure_list(event.get("indicators", []))
        if indicators:
            output += fetch_indicators(indicators, start_date, end_date, api_key)

        # 4. News sentiment
        news_topics = _ensure_list(event.get("news_topics", []))
        if news_topics:
            news_limit = event.get("news_limit", 5)
            output += fetch_news(news_topics, news_limit, api_key)

        if not output:
            output = (
                "No data requested. Provide at least one of:"
                " spot_symbols, historical_symbols,"
                " indicators, or news_topics."
            )

        return {"content": [{"type": "text", "text": output}]}

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": f"Internal server error: {str(e)}"}
