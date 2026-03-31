import os 
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings


class VectorStore:
    """Manages ChromaDB vector database for trade and pattern memories."""

    TRADE_COLLECTION = "trade_memories"
    PATTERN_COLLECTION = "pattern_memories"

    def __init__(self, persist_directory: str = "./data/chromadb"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Create collections with cosine similarity (good for sentence embeddings)
        self.trade_collection = self.client.get_or_create_collection(
            name=self.TRADE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        self.pattern_collection = self.client.get_or_create_collection(
            name=self.PATTERN_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_trade(
        self,
        trade_id: str,
        embedding: List[float],
        metadata: Dict,
        document: str,
    ) -> None:
        """Add a trade memory to the vector store."""
        self.trade_collection.upsert(
            ids=[trade_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document],
        )
    
    def query_trades(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict:
        """Query similar trades from the vector store."""
        kwargs = {
            "query_embeddings":[query_embedding],
            "n_results": n_results,
        }
            
        if where:
            kwargs["where"] = where
        return self.trade_collection.query(**kwargs)

    def add_pattern(
        self,
        pattern_id: str,
        embedding: List[float],
        metadata: Dict,
        document: str,
    ) -> None:
        """Add a pattern memory to the vector store."""
        self.pattern_collection.upsert(
            ids=[pattern_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document],
        )

    def query_patterns(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Dict:
        """Query similar patterns from the vector store."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }

        if where:
            kwargs["where"] = where

        return self.pattern_collection.query(**kwargs)
    
    def get_trade_count(self) -> int:
        """Get total number of trade memories."""
        return self.trade_collection.count()

    def get_pattern_count(self) -> int:
        """Get total number of pattern memories."""
        return self.pattern_collection.count()

    def delete_trade(self, trade_id: str) -> None:
        """Delete a trade memory."""
        self.trade_collection.delete(ids=[trade_id])

    def delete_pattern(self, pattern_id: str) -> None:
        """Delete a pattern memory."""
        self.pattern_collection.delete(ids=[pattern_id])
