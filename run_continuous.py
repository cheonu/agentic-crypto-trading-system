"""Continuous paper trading — runs the pipeline on a schedule.

Designed for deployment on GCP. Runs every N minutes,
logs all decisions to stdout (picked up by Cloud Logging),
and stores results to a local JSON file for analysis.

Usage:
    poetry run python run_continuous.py
    poetry run python run_continuous.py --interval 300 --symbol ETH/USDT

Environment variables:
    OPENAI_API_KEY     — for CrewAI agents
    ANTHROPIC_API_KEY  — for Anthropic-based agents
    GOOGLE_API_KEY     — for Gemini-based agents
    TRADING_INTERVAL   — seconds between cycles (default: 300)
    TRADING_SYMBOL     — symbol to trade (default: BTC/USDT)
    TRADING_MODE       — "langgraph" or "crewai" (default: langgraph)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from paper_trading import (
    fetch_market_data,
    detect_regime,
    estimate_sentiment,
    run_agents,
    run_debate,
    check_risk,
    execute_trade,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("continuous_trading")


# Results storage
RESULTS_FILE = Path("data/trading_results.jsonl")
POSITION_STATE_FILE = Path("data/position_state.json")
RUNNING = True

# Day trading component singletons (initialized in main)
day_trading_config: DayTradingConfig = None  # type: ignore[assignment]
position_manager: PositionManager = None  # type: ignore[assignment]
session_manager: TradingSessionManager = None  # type: ignore[assignment]
intraday_analyzer: IntradayTrendAnalyzer = None  # type: ignore[assignment]
news_provider: NewsSignalProvider = None  # type: ignore[assignment]
stop_loss_monitor: StopLossMonitor = None  # type: ignore[assignment]
day_strategy: DayTradingStrategy = None  # type: ignore[assignment]
fee_filter: FeeAwareFilter = None  # type: ignore[assignment]

# Track previous regime between cycles for transition detection
previous_regime = None


def save_result(result: dict) -> None:
    """Append a result to the JSONL file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def handle_shutdown(signum, frame):
    """Handle graceful shutdown."""
    global RUNNING
    logger.info("Shutdown signal received, finishing current cycle...")
    RUNNING = False


signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)


def init_day_trading_components() -> None:
    """Initialize all day trading components at startup."""
    global day_trading_config, position_manager, session_manager
    global intraday_analyzer, news_provider, stop_loss_monitor
    global day_strategy, fee_filter

    logger.info("Initializing day trading components...")

    day_trading_config = DayTradingConfig()

    position_manager = PositionManager(
        state_filepath=str(POSITION_STATE_FILE),
    )

    session_manager = TradingSessionManager()

    intraday_analyzer = IntradayTrendAnalyzer(
        short_ema_period=9,
        long_ema_period=21,
        rsi_period=14,
        vwap_enabled=True,
    )

    news_provider = NewsSignalProvider(
        cache_ttl_minutes=day_trading_config.news_cache_ttl_minutes,
    )

    stop_loss_monitor = StopLossMonitor(
        position_manager=position_manager,
        default_stop_loss_pct=day_trading_config.stop_loss_pct,
        trailing_stop_pct=day_trading_config.trailing_stop_pct,
    )

    day_strategy = DayTradingStrategy(
        config=day_trading_config,
        session_manager=session_manager,
    )

    fee_filter = FeeAwareFilter(min_profit_fee_ratio=1.5)  # lower for scalping

    logger.info("Day trading components initialized successfully")


async def run_cycle(symbol: str, cycle_num: int) -> dict:
    """Run one trading cycle with day trading integration.

    Pipeline order:
    1. fetch_market_data (existing)
    2. IntradayTrendAnalyzer.analyze (NEW)
    3. detect_regime (existing)
    4. NewsSignalProvider.get_news_signal (NEW)
    5. estimate_sentiment (existing)
    6. StopLossMonitor.check_stop_losses + update_trailing_stop (NEW)
    7. DayTradingStrategy.evaluate (NEW)
    8. FeeAwareFilter.filter_signal (NEW)
    9. If not HOLD: run_agents → run_debate → check_risk → execute_trade
    10. PositionManager update on execution (NEW)
    """
    global previous_regime

    start = time.time()
    logger.info(f"=== Cycle {cycle_num} starting for {symbol} ===")

    try:
        # 1. Fetch market data (existing)
        market_data = fetch_market_data(symbol)

        # 2. IntradayTrendAnalyzer (NEW)
        intraday_data = {
            "candles": market_data.get("candles", []),
            "current_price": market_data.get("price", 0.0),
        }
        intraday_signals = intraday_analyzer.analyze(intraday_data)
        logger.info(
            f"  Intraday: trend={intraday_signals.trend}, "
            f"momentum={intraday_signals.momentum:.2f}, "
            f"rsi={intraday_signals.rsi:.1f}, "
            f"ema_cross={intraday_signals.ema_cross}"
        )

        # 3. Detect regime (existing)
        regime = detect_regime(market_data)

        # 4. NewsSignalProvider (NEW)
        base_symbol = symbol.split("/")[0]
        news_signal = news_provider.get_news_signal(base_symbol)
        logger.info(
            f"  News: score={news_signal.score:.2f}, "
            f"headlines={news_signal.headline_count}, "
            f"events={news_signal.event_flags}"
        )

        # 5. Estimate sentiment (existing)
        sentiment = estimate_sentiment(market_data)

        # 6. StopLossMonitor: check stop-losses and handle emergency sells (NEW)
        stop_loss_market_data = {"prices": {base_symbol: market_data["price"]}}
        stop_loss_events = stop_loss_monitor.check_stop_losses(stop_loss_market_data)

        for event in stop_loss_events:
            logger.warning(
                f"  EMERGENCY {event.exit_reason.upper()}: {event.symbol} "
                f"triggered at ${event.trigger_price:,.2f} "
                f"(entry: ${event.entry_price:,.2f}, loss: {event.loss_pct:.2%})"
            )
            position_manager.close_position(
                event.symbol, event.trigger_price, exit_reason=event.exit_reason
            )

        # Update trailing stops for remaining open positions
        for sym, pos in position_manager.get_all_positions().items():
            current_price = stop_loss_market_data["prices"].get(sym)
            if current_price is not None:
                stop_loss_monitor.update_trailing_stop(sym, current_price)

        # 7. DayTradingStrategy: evaluate trade signal (NEW)
        trade_signal = day_strategy.evaluate(
            current_regime=regime,
            previous_regime=previous_regime,
            sentiment={"score": sentiment["score"], "news_score": news_signal.score},
            intraday_signals=intraday_signals,
            position_manager=position_manager,
            symbol=base_symbol,
        )
        logger.info(
            f"  Strategy: {trade_signal.action} "
            f"(confidence={trade_signal.confidence:.2f}, "
            f"reason={trade_signal.reason})"
        )

        # Store regime for next cycle's transition detection
        previous_regime = regime

        # 8. FeeAwareFilter (NEW)
        # All-in: use full portfolio value to calculate trade size
        price = market_data["price"]
        trade_size = round(day_trading_config.portfolio_value / price, 6)
        # Ensure minimum trade size for Binance (0.00001 BTC)
        trade_size = max(trade_size, 0.00001)

        filtered_signal = fee_filter.filter_signal(
            trade_signal, trade_size, price
        )
        if filtered_signal.action != trade_signal.action:
            logger.info(
                f"  FeeFilter: {trade_signal.action} → {filtered_signal.action} "
                f"({filtered_signal.reason})"
            )

        # 9. Existing pipeline: only run if signal is not HOLD
        execution = {"executed": False}
        debate_result = {"position": "HOLD", "confidence": 0.0, "rounds": 0, "reasoning": ""}
        risk_result = {"approved": False}

        if filtered_signal.action != "HOLD":
            agent_results = await run_agents(market_data, regime, sentiment)
            debate_result = run_debate(agent_results, regime, sentiment)
            risk_result = check_risk(debate_result, market_data)
            execution = execute_trade(risk_result, market_data)
        else:
            logger.info("  Pipeline: Skipping agents/debate/risk/execute (HOLD)")

        # 10. Update PositionManager after trade execution (NEW)
        if execution.get("executed"):
            exec_price = execution.get("price", market_data["price"])
            exec_side = execution.get("side", "")

            if exec_side == "buy" and not position_manager.has_open_position(base_symbol):
                stop_loss_price = exec_price * (1 - day_trading_config.stop_loss_pct)
                take_profit_price = exec_price * (1 + day_trading_config.take_profit_pct)
                position_manager.open_position(
                    symbol=base_symbol,
                    side="long",
                    entry_price=exec_price,
                    size=trade_size,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                )
                logger.info(
                    f"  Position opened: LONG {base_symbol} @ ${exec_price:,.2f} "
                    f"(SL: ${stop_loss_price:,.2f}, TP: ${take_profit_price:,.2f})"
                )
            elif exec_side == "sell" and position_manager.has_open_position(base_symbol):
                closed = position_manager.close_position(
                    base_symbol, exec_price, exit_reason="regime_change"
                )
                logger.info(
                    f"  Position closed: {base_symbol} @ ${exec_price:,.2f} "
                    f"(P&L: ${closed.realized_pnl:,.2f}, {closed.realized_pnl_pct:.2%})"
                )

        elapsed = time.time() - start

        # --- P&L Performance Summary ---
        trade_history = position_manager.get_trade_history()
        total_trades = len(trade_history)
        wins = [t for t in trade_history if t.realized_pnl > 0]
        losses = [t for t in trade_history if t.realized_pnl <= 0]
        total_pnl = sum(t.realized_pnl for t in trade_history)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = (sum(t.realized_pnl for t in wins) / len(wins)) if wins else 0
        avg_loss = (sum(t.realized_pnl for t in losses) / len(losses)) if losses else 0

        # Unrealized P&L from open positions
        unrealized_pnl = 0.0
        for sym, pos in position_manager.get_all_positions().items():
            current = market_data["price"] if sym == base_symbol else 0
            if current > 0:
                if pos.side == "long":
                    unrealized_pnl += (current - pos.entry_price) * pos.size
                else:
                    unrealized_pnl += (pos.entry_price - current) * pos.size

        account_value = day_trading_config.portfolio_value + total_pnl + unrealized_pnl

        logger.info(
            f"  📊 P&L: realized=${total_pnl:+.2f} | unrealized=${unrealized_pnl:+.2f} | "
            f"account=${account_value:.2f} | trades={total_trades} | "
            f"win_rate={win_rate:.0f}% | avg_win=${avg_win:+.2f} | avg_loss=${avg_loss:+.2f}"
        )

        result = {
            "cycle": cycle_num,
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "price": market_data["price"],
            "regime": regime["regime"],
            "regime_confidence": regime["confidence"],
            "sentiment": sentiment["score"],
            "intraday_trend": intraday_signals.trend,
            "intraday_momentum": intraday_signals.momentum,
            "news_score": news_signal.score,
            "trade_signal": filtered_signal.action,
            "trade_confidence": filtered_signal.confidence,
            "trade_reason": filtered_signal.reason,
            "stop_loss_events": len(stop_loss_events),
            "debate_position": debate_result.get("position", "HOLD"),
            "debate_confidence": debate_result.get("confidence", 0.0),
            "risk_approved": risk_result.get("approved", False),
            "executed": execution.get("executed", False),
            "execution_side": execution.get("side", ""),
            "execution_price": execution.get("price", 0),
            "has_open_position": position_manager.has_open_position(base_symbol),
            "total_pnl": round(total_pnl, 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "account_value": round(account_value, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 1),
            "elapsed_seconds": round(elapsed, 2),
            "mode": os.getenv("TRADING_MODE", "langgraph"),
        }

        save_result(result)
        logger.info(
            f"Cycle {cycle_num} complete in {elapsed:.1f}s: "
            f"signal={filtered_signal.action} @ ${market_data['price']:,.2f} "
            f"({'executed' if execution.get('executed') else 'skipped'})"
        )
        return result

    except Exception as e:
        logger.error(f"Cycle {cycle_num} failed: {e}", exc_info=True)
        return {"cycle": cycle_num, "error": str(e)}


async def main():
    """Main loop — run trading cycles on schedule."""
    interval = int(os.getenv("TRADING_INTERVAL", "300"))  # 5 min default
    symbol = os.getenv("TRADING_SYMBOL", "BTC/USDT")
    mode = os.getenv("TRADING_MODE", "langgraph")

    # Initialize day trading components
    init_day_trading_components()

    logger.info(f"Starting continuous paper trading")
    logger.info(f"  Symbol:   {symbol}")
    logger.info(f"  Interval: {interval}s")
    logger.info(f"  Mode:     {mode}")
    logger.info(f"  Results:  {RESULTS_FILE}")
    logger.info(f"  Positions: {POSITION_STATE_FILE}")

    cycle = 0
    while RUNNING:
        cycle += 1
        await run_cycle(symbol, cycle)

        if not RUNNING:
            break

        logger.info(f"Sleeping {interval}s until next cycle...")
        # Sleep in small chunks so we can respond to shutdown signals
        for _ in range(interval):
            if not RUNNING:
                break
            await asyncio.sleep(1)

    logger.info(f"Stopped after {cycle} cycles. Results in {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
