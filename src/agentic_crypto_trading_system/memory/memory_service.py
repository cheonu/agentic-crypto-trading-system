import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from .vector_store import VectorStore


class MemoryService:
    """High-level memory service for storing and querying trade and pattern memories."""

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.vector_store = vector_store
        self.model = SentenceTransformer(model_name)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector from text."""
        return self.model.encode(text).tolist()

    def _build_trade_document(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        regime: str,
        agent_id: str,       
    ) -> str:
        """Build a text document describing a trade for embedding."""
        outcome = "profit" if pnl > 0 else "loss"
        return(
            f"{direction} {symbol} trade during {regime} regime. "
            f"Entry {entry_price}, Exit {exit_price}. "
            f"Result: {outcome} of {abs(pnl_pct):.2f}%. "
            f"Agent: {agent_id}."
        )

    def store_trade_outcome(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        regime: str,
        agent_id: str,
        duration_seconds: int = 0,
    ):
        """Store a trade outcome with its context."""
        document = self._build_trade_document (
            symbol, direction, entry_price, exit_price, pnl, pnl_pct, regime, agent_id
        )
        embedding = self._generate_embedding(document)
        metadata = {
           "symbol": symbol,
            "direction": direction,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "regime": regime,
            "agent_id": agent_id,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.utcnow().isoformat(), 
        }
        self.vector_store.add_trade(trade_id, embedding, metadata, document)

    def store_pattern(
       self,
        pattern_id: str,
        description: str,
        pattern_type: str,
        symbol: str,
        regime: str,
        confidence: float,
        agent_id: str, 
    ) -> None:
        """Store a discovered pattern in memory. """
        document = (
            f"{pattern_type} pattern on {symbol} during {regime} regime: "
            f"{description}. Confidence: {confidence:.2f})"
        )
        embedding = self._generate_embedding(document)
        metadata = {
            "pattern_type": pattern_type,
            "symbol": symbol,
            "regime": regime,
            "confidence": float(confidence),
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.vector_store.add_pattern(pattern_id, embedding, metadata, document)

    def query_similar_trades(
        self,
        query_text: str,
        n_results: int = 10,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Dict:
        """Query similar past trades using semantic search."""
        embedding = self._generate_embedding(query_text)
        where = {}
        if symbol:
            where["symbol"] = symbol
        if regime:
            where["regime"] = regime

        return self.vector_store.query_trades(
            query_embedding = embedding, 
            n_results = n_results, 
            where = where if where else None,
        )
        
    def query_patterns(
        self,
        query_text: str,
        n_results: int = 10,
        regime: Optional[str] = None,
        pattern_type: Optional[str] = None,
    ) -> Dict:
        """Query similar patterns with optional regime filtering."""

        embedding = self._generate_embedding(query_text)
        where = {}
        if regime:
            where["regime"] = regime
        if pattern_type:
            where["pattern_type"] = pattern_type

        return self.vector_store.query_patterns(
            query_embedding=embedding,
            n_results=n_results,
            where=where if where else None,
        )
