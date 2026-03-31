from agentic_crypto_trading_system.config import load_config

def test_config_loads():
    """Test that config loads successfully."""
    config = load_config()
    assert config.app["name"] == "Agentic Crypto Trading System"
    assert config.risk.max_leverage > 0
