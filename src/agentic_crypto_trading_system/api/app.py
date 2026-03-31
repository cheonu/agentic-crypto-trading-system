"""FastAPI application — REST and WebSocket endpoints.

Provides API for querying positions, agent performance,
controlling agents, updating risk limits, and emergency stop.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# --- Request/Response Models ---

class RiskLimitsUpdate(BaseModel):
    max_position_size: Optional[float] = None
    max_portfolio_exposure: Optional[float] = None
    max_daily_loss: Optional[float] = None
    max_leverage: Optional[float] = None


class AgentAction(BaseModel):
    action: str  # "start" or "stop"


class EmergencyStopRequest(BaseModel):
    reason: str = "manual"


class PositionResponse(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float


class AgentPerformanceResponse(BaseModel):
    agent_role: str
    sharpe_ratio: float
    win_rate: float
    trades_count: int
    is_active: bool


# --- WebSocket Manager ---

class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""

    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.setdefault(channel, []).append(websocket)

    def disconnect(self, channel: str, websocket: WebSocket) -> None:
        if channel in self.connections:
            self.connections[channel] = [
                ws for ws in self.connections[channel] if ws != websocket
            ]

    async def broadcast(self, channel: str, data: Dict[str, Any]) -> None:
        dead = []
        for ws in self.connections.get(channel, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(channel, ws)


# --- App Factory ---

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic Crypto Trading System",
        version="1.0.0",
        description="Multi-agent cryptocurrency trading platform",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ws_manager = ConnectionManager()

    # --- REST Endpoints ---

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/positions")
    async def get_positions():
        # Placeholder — wire to actual position tracking
        return {"positions": []}

    @app.get("/agents/performance")
    async def get_agent_performance():
        return {"agents": []}

    @app.post("/agents/{agent_id}/start")
    async def start_agent(agent_id: str):
        logger.info(f"Starting agent: {agent_id}")
        return {"agent_id": agent_id, "status": "started"}

    @app.post("/agents/{agent_id}/stop")
    async def stop_agent(agent_id: str):
        logger.info(f"Stopping agent: {agent_id}")
        return {"agent_id": agent_id, "status": "stopped"}

    @app.put("/risk/limits")
    async def update_risk_limits(limits: RiskLimitsUpdate):
        logger.info(f"Risk limits updated: {limits.model_dump()}")
        return {"status": "updated", "limits": limits.model_dump()}

    @app.get("/memory/trades")
    async def query_trade_memories():
        return {"trades": []}

    @app.get("/memory/patterns")
    async def query_pattern_memories():
        return {"patterns": []}

    @app.post("/emergency-stop")
    async def emergency_stop(request: EmergencyStopRequest):
        logger.critical(f"Emergency stop triggered: {request.reason}")
        return {"status": "halted", "reason": request.reason}

    # --- WebSocket Endpoints ---

    @app.websocket("/ws/trades")
    async def ws_trades(websocket: WebSocket):
        await ws_manager.connect("trades", websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect("trades", websocket)

    @app.websocket("/ws/positions")
    async def ws_positions(websocket: WebSocket):
        await ws_manager.connect("positions", websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect("positions", websocket)

    @app.websocket("/ws/regimes")
    async def ws_regimes(websocket: WebSocket):
        await ws_manager.connect("regimes", websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect("regimes", websocket)

    @app.websocket("/ws/debates")
    async def ws_debates(websocket: WebSocket):
        await ws_manager.connect("debates", websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect("debates", websocket)

    return app
