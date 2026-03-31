import os
from pathlib import Path
from typing import Any, Dict
import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseModel):
    """Database configuration. """
    postgres: Dict[str, Any]
    redis: Dict[str, Any]

class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_position_size_usd: float
    max_position_size_pct: float
    max_portfolio_exposure_pct: float
    max_daily_loss_usd: float
    max_daily_loss_pct: float
    max_leverage: float
    var_limit_95: float
    stop_loss_pct: float

class Config(BaseModel):
    """Main application configuration."""
    app: Dict[str, Any]
    database: DatabaseConfig
    vector_db: Dict[str, Any]
    exchanges: Dict[str, Any]
    risk: RiskConfig
    agents: Dict[str, Any]
    api: Dict[str, Any]
    logging: Dict[str, Any]

    class ConfigDict:
        env_file = ".env"
        env_file_encoding = "utf-8"

def load_config(config_path: str = None, environment: str = None) -> Config:
    """
    Load configuration from YAML files.
    
    Args:
        config_path: Path to config file (default: config/default.yaml)
        environment: Environment name (development, production, etc.)
    
    Returns:
        Config object with loaded settings
    """
    if config_path is None:
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_path = config_dir / "default.yaml"
    
    # Load default config
    with open (config_path) as f:
        config_data = yaml.safe_load(f)

    # Load environment-specific overrides
    if environment:
        env_config_path = Path(config_path).parent / f"{environment}.yaml"
        if env_config_path.exists():
            with open(env_config_path) as f:
                env_data = yaml.safe_load(f)
                # Deep merge configs
                config_data = deep_merge(config_data, env_data)

    # override with environmental variable
    env = os.getenv("ENVIRONMENT", "development")
    if env!= "development":
        config_data["app"]["environment"] = env
    return Config(**config_data)

def deep_merge(base: dict, override:dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

# Global config instance
_config: Config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        env = os.getenv("ENVIRONMENT", "development")
        _config = load_config(environment=env)
    return _config
