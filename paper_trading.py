"""Paper Trading — full pipeline on real market data.

Supports two modes:
- PAPER: Simulated execution (no exchange account needed)
- TESTNET: Real orders on Binance testnet (free fake money)

Set TRADING_TESTNET=true and provide BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET
to enable testnet mode.

Usage:
    poetry run python paper_trading.py

Runs one analysis cycle. For continuous trading, you'd put this in a loop.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import ccxt

from agentic_crypto_trading_system.agents.base import (
    AgentConfig, AgentRole, Task, TaskResult,
)
from agentic_crypto_trading_system.agents.langgraph_framework import (
    LangGraphFramework,
)
from agentic_crypto_trading_system.day_trading import (
    DayTradingConfig,
    DayTradingStrategy,
    FeeAwareFilter,
    IntradayTrendAnalyzer,
    NewsSignalProvider,
    PositionManager,
    StopLossMonitor,
    TradingSessionManager,
)
from agentic_crypto_trading_system.debate import (
    DebateService, DebateArgument, Position, ConsensusMode,
)
from agentic_crypto_trading_system.risk.manager import RiskManager, TradeProposal
from agentic_crypto_trading_system.risk.limits import RiskLimits
from agentic_crypto_trading_system.state.manager import (
    StateManager, SystemMode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_trading")

PAPER_POSITION_STATE_FILE = Path("data/paper_position_state.json")


# ─── Exchange Setup ───

def get_exchange() -> ccxt.binance:
    """Create exchange instance. Uses testnet if configured."""
    use_testnet = os.getenv("TRADING_TESTNET", "false").lower() == "true"
    api_key = os.getenv("BINANCE_TESTNET_KEY", "")
    api_secret = os.getenv("BINANCE_TESTNET_SECRET", "")

    config = {"enableRateLimit": True}

    if use_testnet and api_key and api_secret:
        config["apiKey"] = api_key
        config["secret"] = api_secret
        config["sandbox"] = True  # This switches to testnet URLs
        logger.info("Using Binance TESTNET (sandbox mode)")
    else:
        logger.info("Using Binance public API (read-only)")

    return ccxt.binance(config)


# ─── Step 1: Fetch live market data ───

def fetch_market_data(symbol: str = "BTC/USDT") -> Dict:
    """Fetch real market data from Binance."""
    exchange = get_exchange()

    logger.info(f"Fetching market data for {symbol}...")

    # Get ticker
    ticker = exchange.fetch_ticker(symbol)

    # Get recent OHLCV (1h candles, last 20)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=20)

    # Calculate simple indicators from OHLCV
    closes = [c[4] for c in ohlcv]
    volumes = [c[5] for c in ohlcv]

    avg_price = sum(closes) / len(closes)
    price_change = (closes[-1] - closes[0]) / closes[0] * 100
    avg_volume = sum(volumes) / len(volumes)
    volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

    # Simple volatility (std of returns)
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 if returns else 0

    # Fetch 5m candles for intraday analysis
    ohlcv_5m = exchange.fetch_ohlcv(symbol, timeframe="5m", limit=50)
    candles_5m = [
        {
            "open": c[1],
            "high": c[2],
            "low": c[3],
            "close": c[4],
            "volume": c[5],
        }
        for c in ohlcv_5m
    ]

    data = {
        "symbol": symbol,
        "price": ticker["last"],
        "bid": ticker["bid"],
        "ask": ticker["ask"],
        "change_24h": ticker.get("percentage", 0),
        "volume_24h": ticker.get("quoteVolume", 0),
        "high_24h": ticker.get("high", 0),
        "low_24h": ticker.get("low", 0),
        "closes": closes,
        "candles": candles_5m,
        "avg_price_20h": round(avg_price, 2),
        "price_change_20h": round(price_change, 2),
        "volume_ratio": round(volume_ratio, 2),
        "volatility": round(volatility, 6),
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(
        f"  Price: ${data['price']:,.2f} | "
        f"24h: {data['change_24h']:+.1f}% | "
        f"Vol ratio: {data['volume_ratio']:.1f}x | "
        f"Volatility: {data['volatility']:.4f}"
    )
    return data


# ─── Step 2: Detect market regime ───

def detect_regime(market_data: Dict) -> Dict:
    """Simple regime detection from market data."""
    change = market_data.get("change_24h", 0) or 0
    volatility = market_data.get("volatility", 0)
    price = market_data["price"]
    avg = market_data["avg_price_20h"]

    # Use env var to control sensitivity (lower = more trades for testing)
    threshold = float(os.getenv("REGIME_THRESHOLD", "0.3"))

    # Classification rules
    if volatility > 0.02:
        regime = "high_volatility"
        confidence = min(0.95, 0.5 + volatility * 10)
    elif change > threshold and price > avg:
        regime = "bull"
        confidence = min(0.95, 0.5 + abs(change) / 10)
    elif change < -threshold and price < avg:
        regime = "bear"
        confidence = min(0.95, 0.5 + abs(change) / 10)
    else:
        regime = "sideways"
        confidence = 0.6

    result = {"regime": regime, "confidence": round(confidence, 2)}
    logger.info(f"  Regime: {regime} (confidence: {confidence:.0%})")
    return result


# ─── Step 3: Sentiment (simplified — no API call) ───

def estimate_sentiment(market_data: Dict) -> Dict:
    """Estimate sentiment from price action (no external API needed)."""
    change = market_data.get("change_24h", 0) or 0
    vol_ratio = market_data.get("volume_ratio", 1)

    # Price momentum as sentiment proxy
    score = max(-1.0, min(1.0, change / 10))

    # High volume amplifies sentiment
    if vol_ratio > 1.5:
        score *= 1.2
        score = max(-1.0, min(1.0, score))

    trend = "improving" if score > 0.05 else "declining" if score < -0.05 else "neutral"

    result = {"score": round(score, 2), "trend": trend}
    logger.info(f"  Sentiment: {score:.2f} ({trend})")
    return result


# ─── Step 4: Run LangGraph agents ───

async def run_agents(market_data: Dict, regime: Dict, sentiment: Dict) -> Dict[str, TaskResult]:
    """Run all three LangGraph agents."""
    lg = LangGraphFramework()

    configs = [
        (AgentRole.TECHNICAL_ANALYST, "Tech Analyst", "Analyze technicals"),
        (AgentRole.SENTIMENT_ANALYST, "Sentiment Analyst", "Analyze sentiment"),
        (AgentRole.RISK_ASSESSOR, "Risk Assessor", "Assess risk"),
    ]
    for role, name, goal in configs:
        lg.create_agent(AgentConfig(name=name, role=role, goal=goal, backstory="Expert"))

    context = {
        "market_data": market_data,
        "sentiment_data": sentiment,
        "regime_data": regime,
    }

    results = {}
    for role, name, _ in configs:
        task = Task(
            description=(
                f"Analyze {market_data['symbol']} at ${market_data['price']:,.2f}. "
                f"Regime: {regime['regime']}. Sentiment: {sentiment['score']}."
            ),
            agent_role=role,
            context=context,
            expected_output="BUY, SELL, or HOLD",
        )
        result = await lg.execute_task(task)
        results[role.value] = result
        logger.info(f"  {name}: {result.output}")

    return results


# ─── Step 5: Debate ───

def run_debate(agent_results: Dict[str, TaskResult], regime: Dict, sentiment: Dict) -> Dict:
    """Run a multi-agent debate based on agent results."""

    def make_generator(role: str, result: TaskResult):
        def gen(r, task, round_num, prev):
            # Map agent output to a Position
            output = result.output.upper()
            if "BUY" in output:
                pos = Position.BUY
            elif "SELL" in output:
                pos = Position.SELL
            else:
                pos = Position.HOLD

            return DebateArgument(
                agent_role=role,
                position=pos,
                confidence=0.75 if result.success else 0.3,
                reasoning=result.reasoning[:200],
            )
        return gen

    generators = {
        role: make_generator(role, result)
        for role, result in agent_results.items()
    }

    debate = DebateService(
        max_rounds=3,
        consensus_mode=ConsensusMode.MAJORITY,
        min_confidence=0.5,
        veto_roles=["risk_assessor"],
    )

    roles = list(agent_results.keys())
    transcript = debate.run_debate("BTC/USDT", "Should we trade?", roles, generators)

    logger.info(
        f"  Debate: {transcript.status.value} -> "
        f"{transcript.final_position.value if transcript.final_position else 'NONE'} "
        f"(confidence: {transcript.final_confidence:.0%}, "
        f"rounds: {len(transcript.rounds)})"
    )

    return {
        "status": transcript.status.value,
        "position": transcript.final_position.value if transcript.final_position else "HOLD",
        "confidence": transcript.final_confidence,
        "rounds": len(transcript.rounds),
        "reasoning": transcript.final_reasoning[:300],
    }


# ─── Step 6: Risk check ───

def check_risk(debate_result: Dict, market_data: Dict) -> Dict:
    """Validate the trade through the risk manager."""
    limits = RiskLimits()
    rm = RiskManager(limits=limits, portfolio_value=100000.0)

    position = debate_result["position"]
    if position == "HOLD":
        logger.info("  Risk: No trade to validate (HOLD)")
        return {"approved": False, "reason": "HOLD — no trade"}

    price = market_data["price"]
    stop_loss = price * 0.98 if position == "BUY" else price * 1.02

    # Use 1% of portfolio for each trade
    trade_value = rm.portfolio_value * 0.01
    size = trade_value / price

    proposal = TradeProposal(
        symbol=market_data["symbol"],
        direction="long" if position == "BUY" else "short",
        size=size,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=price * 1.06 if position == "BUY" else price * 0.94,
    )

    result = rm.validate_trade(proposal)

    logger.info(f"  Risk: {'APPROVED' if result.approved else 'REJECTED'} for {position}")
    if not result.approved:
        logger.info(f"  Reasons: {result.reasons}")

    return {
        "approved": result.approved,
        "trade": {
            "symbol": market_data["symbol"],
            "side": "buy" if position == "BUY" else "sell",
            "size": proposal.size,
            "price": price,
        },
        "reasons": result.reasons,
    }


# ─── Step 7: Execute trade ───

def execute_trade(risk_result: Dict, market_data: Dict) -> Dict:
    """Execute trade — paper simulation or real testnet order."""
    if not risk_result["approved"]:
        logger.info("  Execution: Skipped (not approved)")
        return {"executed": False}

    trade = risk_result["trade"]
    use_testnet = os.getenv("TRADING_TESTNET", "false").lower() == "true"

    if use_testnet:
        return _execute_testnet(trade, market_data)
    else:
        return _execute_paper(trade, market_data)


def _execute_testnet(trade: Dict, market_data: Dict) -> Dict:
    """Place a real order on Binance testnet."""
    exchange = get_exchange()
    symbol = trade["symbol"]
    side = trade["side"]
    price = market_data["price"]

    # Use $15 worth of the asset to stay above Binance minimum notional ($5-10)
    amount = round(15.0 / price, 6) if price > 0 else 0.001

    try:
        # Place market order on testnet
        order = exchange.create_market_order(symbol, side, amount)

        result = {
            "executed": True,
            "mode": "testnet",
            "order_id": order.get("id", ""),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": order.get("average", price),
            "cost": order.get("cost", 0),
            "fee": order.get("fee", {}).get("cost", 0),
            "status": order.get("status", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"  TESTNET ORDER: {side.upper()} {amount} {symbol} "
            f"@ ${result['price']:,.2f} | "
            f"Order ID: {result['order_id']} | "
            f"Status: {result['status']}"
        )
        return result

    except Exception as e:
        logger.error(f"  Testnet order failed: {e}")
        return {"executed": False, "error": str(e)}


def _execute_paper(trade: Dict, market_data: Dict) -> Dict:
    """Simulate trade execution (paper mode)."""
    price = market_data["price"]
    slippage = price * 0.0005
    exec_price = price + slippage if trade["side"] == "buy" else price - slippage
    commission = exec_price * trade["size"] * 0.001

    result = {
        "executed": True,
        "mode": "paper",
        "symbol": trade["symbol"],
        "side": trade["side"],
        "price": round(exec_price, 2),
        "size_pct": trade["size"],
        "commission": round(commission, 2),
        "slippage": round(slippage, 2),
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(
        f"  PAPER TRADE: {trade['side'].upper()} {trade['symbol']} "
        f"@ ${exec_price:,.2f} (size: {trade['size']:.0%}, "
        f"commission: ${commission:.2f})"
    )
    return result


# ─── Main Pipeline ───

async def run_pipeline(symbol: str = "BTC/USDT"):
    """Run the full paper trading pipeline once with day trading integration.

    Pipeline order mirrors run_continuous.py:
    1. Initialize day trading components (single-cycle, no module-level singletons)
    2. fetch_market_data
    3. IntradayTrendAnalyzer.analyze
    4. detect_regime
    5. NewsSignalProvider.get_news_signal
    6. estimate_sentiment
    7. StopLossMonitor.check_stop_losses + update_trailing_stop
    8. DayTradingStrategy.evaluate
    9. FeeAwareFilter.filter_signal
    10. If not HOLD: run_agents → run_debate → check_risk → execute_trade
    11. PositionManager update on execution
    """
    print("\n" + "=" * 60)
    print(f"  PAPER TRADING PIPELINE — {symbol}")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    # Initialize day trading components for this single-cycle run
    day_config = DayTradingConfig()
    position_manager = PositionManager(state_filepath=str(PAPER_POSITION_STATE_FILE))
    session_manager = TradingSessionManager()
    intraday_analyzer = IntradayTrendAnalyzer(
        short_ema_period=9, long_ema_period=21, rsi_period=14, vwap_enabled=True,
    )
    news_provider = NewsSignalProvider(
        cache_ttl_minutes=day_config.news_cache_ttl_minutes,
    )
    stop_loss_monitor = StopLossMonitor(
        position_manager=position_manager,
        default_stop_loss_pct=day_config.stop_loss_pct,
        trailing_stop_pct=day_config.trailing_stop_pct,
    )
    day_strategy = DayTradingStrategy(config=day_config, session_manager=session_manager)
    fee_filter = FeeAwareFilter()

    base_symbol = symbol.split("/")[0]

    # Step 1: Market data
    print("\n📊 Step 1: Fetching market data...")
    market_data = fetch_market_data(symbol)

    # Step 2: Intraday trend analysis (NEW)
    print("\n📈 Step 2: Intraday trend analysis...")
    intraday_data = {
        "candles": market_data.get("candles", []),
        "current_price": market_data.get("price", 0.0),
    }
    intraday_signals = intraday_analyzer.analyze(intraday_data)
    logger.info(
        "  Intraday: trend=%s, momentum=%.2f, rsi=%.1f, ema_cross=%s",
        intraday_signals.trend, intraday_signals.momentum,
        intraday_signals.rsi, intraday_signals.ema_cross,
    )

    # Step 3: Regime detection
    print("\n🔍 Step 3: Detecting market regime...")
    regime = detect_regime(market_data)

    # Step 4: News signal (NEW)
    print("\n📰 Step 4: Fetching news signal...")
    news_signal = news_provider.get_news_signal(base_symbol)
    logger.info(
        "  News: score=%.2f, headlines=%d, events=%s",
        news_signal.score, news_signal.headline_count, news_signal.event_flags,
    )

    # Step 5: Sentiment
    print("\n💭 Step 5: Estimating sentiment...")
    sentiment = estimate_sentiment(market_data)

    # Step 6: Stop-loss monitoring (NEW)
    print("\n🛑 Step 6: Checking stop-losses...")
    stop_loss_market_data = {"prices": {base_symbol: market_data["price"]}}
    stop_loss_events = stop_loss_monitor.check_stop_losses(stop_loss_market_data)

    for event in stop_loss_events:
        logger.warning(
            "  EMERGENCY %s: %s triggered at $%,.2f "
            "(entry: $%,.2f, loss: %.2%%)",
            event.exit_reason.upper(), event.symbol,
            event.trigger_price, event.entry_price, event.loss_pct,
        )
        position_manager.close_position(
            event.symbol, event.trigger_price, exit_reason=event.exit_reason,
        )

    # Update trailing stops for remaining open positions
    for sym, pos in position_manager.get_all_positions().items():
        current_price = stop_loss_market_data["prices"].get(sym)
        if current_price is not None:
            stop_loss_monitor.update_trailing_stop(sym, current_price)

    # Step 7: Day trading strategy evaluation (NEW)
    print("\n🎯 Step 7: Evaluating day trading strategy...")
    trade_signal = day_strategy.evaluate(
        current_regime=regime,
        previous_regime=None,  # single-cycle run, no previous regime
        sentiment={"score": sentiment["score"], "news_score": news_signal.score},
        intraday_signals=intraday_signals,
        position_manager=position_manager,
        symbol=base_symbol,
    )
    logger.info(
        "  Strategy: %s (confidence=%.2f, reason=%s)",
        trade_signal.action, trade_signal.confidence, trade_signal.reason,
    )

    # Step 8: Fee-aware filter (NEW)
    print("\n💲 Step 8: Fee-aware filtering...")
    trade_size = 0.001  # default trade size in base currency
    filtered_signal = fee_filter.filter_signal(
        trade_signal, trade_size, market_data["price"],
    )
    if filtered_signal.action != trade_signal.action:
        logger.info(
            "  FeeFilter: %s → %s (%s)",
            trade_signal.action, filtered_signal.action, filtered_signal.reason,
        )

    # Steps 9-12: Existing pipeline — only run if signal is not HOLD
    execution = {"executed": False}
    debate_result = {"position": "HOLD", "confidence": 0.0, "rounds": 0, "reasoning": ""}
    risk_result = {"approved": False}

    if filtered_signal.action != "HOLD":
        print("\n🤖 Step 9: Running LangGraph agents...")
        agent_results = await run_agents(market_data, regime, sentiment)

        print("\n🗣️  Step 10: Multi-agent debate...")
        debate_result = run_debate(agent_results, regime, sentiment)

        print("\n🛡️  Step 11: Risk validation...")
        risk_result = check_risk(debate_result, market_data)

        print("\n💰 Step 12: Execution...")
        execution = execute_trade(risk_result, market_data)
    else:
        print("\n⏸️  Steps 9-12: Skipped (HOLD signal)")

    # Update PositionManager after trade execution (NEW)
    if execution.get("executed"):
        exec_price = execution.get("price", market_data["price"])
        exec_side = execution.get("side", "")

        if exec_side == "buy" and not position_manager.has_open_position(base_symbol):
            stop_loss_price = exec_price * (1 - day_config.stop_loss_pct)
            take_profit_price = exec_price * (1 + day_config.take_profit_pct)
            position_manager.open_position(
                symbol=base_symbol,
                side="long",
                entry_price=exec_price,
                size=trade_size,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
            logger.info(
                "  Position opened: LONG %s @ $%,.2f (SL: $%,.2f, TP: $%,.2f)",
                base_symbol, exec_price, stop_loss_price, take_profit_price,
            )
        elif exec_side == "sell" and position_manager.has_open_position(base_symbol):
            closed = position_manager.close_position(
                base_symbol, exec_price, exit_reason="regime_change",
            )
            logger.info(
                "  Position closed: %s @ $%,.2f (P&L: $%,.2f, %.2%%)",
                base_symbol, exec_price, closed.realized_pnl, closed.realized_pnl_pct,
            )

    # Summary
    session = session_manager.get_current_session()
    open_positions = position_manager.get_all_positions()

    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Symbol:      {symbol}")
    print(f"  Price:       ${market_data['price']:,.2f}")
    print(f"  Regime:      {regime['regime']} ({regime['confidence']:.0%})")
    print(f"  Sentiment:   {sentiment['score']:+.2f} ({sentiment['trend']})")
    print(f"  Intraday:    {intraday_signals.trend} (momentum: {intraday_signals.momentum:+.2f}, RSI: {intraday_signals.rsi:.1f})")
    print(f"  News:        score={news_signal.score:+.2f}, headlines={news_signal.headline_count}")
    print(f"  Session:     {session.name} (threshold: {session.confidence_threshold:.0%})")
    print(f"  Signal:      {filtered_signal.action} (confidence: {filtered_signal.confidence:.0%})")
    print(f"  Debate:      {debate_result['position']} ({debate_result['confidence']:.0%})")
    print(f"  Risk:        {'APPROVED' if risk_result['approved'] else 'REJECTED'}")
    print(f"  Executed:    {'YES' if execution.get('executed') else 'NO'}")
    if execution.get("executed"):
        print(f"  Trade:       {execution['side'].upper()} @ ${execution['price']:,.2f}")
    print(f"  Stop-Loss:   {len(stop_loss_events)} event(s)")
    print(f"  Open Pos:    {len(open_positions)} position(s)")
    for sym, pos in open_positions.items():
        print(f"    {sym}: {pos.side} @ ${pos.entry_price:,.2f} (SL: ${pos.stop_loss_price:,.2f})")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_pipeline("BTC/USDT"))
