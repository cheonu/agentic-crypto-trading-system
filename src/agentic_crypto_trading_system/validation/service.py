"""Configuration validation — schema checks, consistency, hot-reload.

Validates all configuration types on load and refuses to start
if configuration is invalid. Supports runtime hot-reloading.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a configuration validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigValidator:
    """Validates system configuration."""

    def validate_risk_limits(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate risk limit configuration."""
        errors = []
        warnings = []

        max_pos = config.get("max_position_size", 0)
        if max_pos <= 0 or max_pos > 1.0:
            errors.append(f"max_position_size must be 0-1, got {max_pos}")

        max_exposure = config.get("max_portfolio_exposure", 0)
        if max_exposure <= 0 or max_exposure > 1.0:
            errors.append(f"max_portfolio_exposure must be 0-1, got {max_exposure}")

        max_daily_loss = config.get("max_daily_loss", 0)
        if max_daily_loss <= 0 or max_daily_loss > 1.0:
            errors.append(f"max_daily_loss must be 0-1, got {max_daily_loss}")

        max_leverage = config.get("max_leverage", 1)
        if max_leverage < 1 or max_leverage > 100:
            errors.append(f"max_leverage must be 1-100, got {max_leverage}")

        if max_pos > max_exposure:
            warnings.append("max_position_size > max_portfolio_exposure")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_agent_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate agent configuration."""
        errors = []

        if not config.get("name"):
            errors.append("Agent name is required")
        if not config.get("role"):
            errors.append("Agent role is required")
        if not config.get("goal"):
            errors.append("Agent goal is required")

        max_iter = config.get("max_iterations", 10)
        if max_iter < 1 or max_iter > 100:
            errors.append(f"max_iterations must be 1-100, got {max_iter}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_exchange_credentials(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate exchange credentials are present."""
        errors = []

        if not config.get("api_key"):
            errors.append("Exchange API key is required")
        if not config.get("api_secret"):
            errors.append("Exchange API secret is required")

        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def validate_all(self, full_config: Dict[str, Any]) -> ValidationResult:
        """Validate the entire system configuration."""
        all_errors = []
        all_warnings = []

        # Risk limits
        risk_result = self.validate_risk_limits(full_config.get("risk", {}))
        all_errors.extend(risk_result.errors)
        all_warnings.extend(risk_result.warnings)

        # Agent configs
        for agent_cfg in full_config.get("agents", []):
            agent_result = self.validate_agent_config(agent_cfg)
            all_errors.extend(agent_result.errors)

        # Exchange
        exchange_result = self.validate_exchange_credentials(full_config.get("exchange", {}))
        all_errors.extend(exchange_result.errors)

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )
