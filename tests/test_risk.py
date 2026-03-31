import pytest
from agentic_crypto_trading_system.risk.limits import RiskLimits
from agentic_crypto_trading_system.risk.manager import RiskManager, TradeProposal


def test_risk_limits_valid():
    """Test default risk limits are valid."""
    limits = RiskLimits()
    assert limits.validate() is True


def test_risk_limits_invalid():
    """Test invalid risk limits."""
    limits = RiskLimits(max_position_size_pct=-1)
    assert limits.validate() is False


def test_validate_trade_approved():
    """Test trade within limits is approved."""
    manager = RiskManager(portfolio_value=100000)
    proposal = TradeProposal(
        symbol="BTC/USDT", direction="long",
        size=0.01, entry_price=50000, stop_loss=49000,
        take_profit=53000, agent_id="agent-1",
    )
    result = manager.validate_trade(proposal)
    assert result.approved is True
    assert len(result.reasons) == 0


def test_validate_trade_too_large():
    """Test trade exceeding size limit is rejected."""
    manager = RiskManager(portfolio_value=100000)
    proposal = TradeProposal(
        symbol="BTC/USDT", direction="long",
        size=1.0, entry_price=50000, stop_loss=49000,
    )
    result = manager.validate_trade(proposal)
    assert result.approved is False
    assert any("Trade size" in r for r in result.reasons)


def test_validate_trade_daily_loss_exceeded():
    """Test trade rejected when daily loss limit hit."""
    manager = RiskManager(portfolio_value=100000)
    manager.daily_pnl = -3100  # Over 3% loss
    proposal = TradeProposal(
        symbol="BTC/USDT", direction="long",
        size=0.01, entry_price=50000, stop_loss=49000,
    )
    result = manager.validate_trade(proposal)
    assert result.approved is False
    assert any("Daily loss" in r for r in result.reasons)


def test_calculate_position_size():
    """Test position size calculation."""
    manager = RiskManager(portfolio_value=100000)
    size = manager.calculate_position_size(entry_price=50000, stop_loss=49000)
    assert size > 0
    # Risk amount = 100000 * 2% = 2000, price risk = 1000, size = 2.0
    assert abs(size - 2.0) < 0.01


def test_calculate_var():
    """Test VaR calculation."""
    manager = RiskManager(portfolio_value=100000)
    returns = [-0.02, -0.01, 0.01, 0.02, -0.03, 0.005, -0.015, 0.01, -0.005, 0.02]
    var = manager.calculate_portfolio_var(returns)
    assert var > 0


def test_update_limits():
    """Test updating risk limits with audit log."""
    manager = RiskManager()
    manager.update_limits(max_daily_loss_pct=5.0)
    assert manager.limits.max_daily_loss_pct == 5.0
    assert len(manager.audit_log) == 1
