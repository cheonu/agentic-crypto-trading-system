import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from .vector_store import VectorStore

class MemoryBackupManager:
    """Manages backup and recovery of vector memory data."""

    def __init__(self, vector_store: VectorStore, backup_directory: str = "./data/backups"):

        self.vector_store = vector_store
        self.backup_directory = backup_directory
        os.makedirs(backup_directory, exist_ok=True)

    def _backup_path(self, backup_type: str) -> str:
        """Generate a timestamped backup path."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join (self.backup_directory, f"{backup_type}_{timestamp}")

    def create_incremental_backup(self) -> str:
        """Create an incremental backup of the vector store data."""
        backup_path = self._backup_path("incremental")
        os.makedirs(backup_path, exist_ok=True)

        # Export trade memories
        trade_count = self.vector_store.get_trade_count()
        if trade_count > 0:
            trades = self.vector_store.trade_collection.get(
                include=["embeddings", "metadatas", "documents"]
            )
            with open(os.path.join(backup_path, "trades.json"), "w") as f:
                json.dump(trades, f, default=str)

        # Export news memories
        pattern_count = self.vector_store.get_pattern_count()
        if pattern_count > 0:
            patterns = self.vector_store.pattern_collection.get(
                include=["embeddings", "metadatas", "documents"]
            )
            with open(os.path.join(backup_path, "patterns.json"), "w") as f:
                json.dump(patterns, f, default=str)
        
        # Write manifest
        manifest = {
            "type": "incremental",
            "timestamp": datetime.utcnow().isoformat(),
            "trade_count": trade_count,
            "pattern_count": pattern_count,
        }
        with open(os.path.join(backup_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        return backup_path
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore memory data from a backup."""
        manifest_path = os.path.join(backup_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No manifest found at {backup_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        if manifest["type"] == "full":
            return self._restore_full(backup_path)
        else:
            return self._restore_incremental(backup_path)

    def _restore_full(self, backup_path: str) -> bool:
        """Restore from a full backup."""
        target = self.vector_store.persist_directory
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(backup_path, target, ignore=shutil.ignore_patterns("manifest.json"))
        return True

    def create_full_backup(self) -> str:
        """Create a full backup by copying the entire ChromaDB directory."""
        backup_path = self._backup_path("full")
        shutil.copytree(self.vector_store.persist_directory, backup_path)

        manifest = {
            "type": "full",
            "timestamp": datetime.utcnow().isoformat(),
            "trade_count": self.vector_store.get_trade_count(),
            "pattern_count": self.vector_store.get_pattern_count(),
        }
        with open(os.path.join(backup_path, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        return backup_path


    def _restore_incremental(self, backup_path: str) -> bool:
        """Restore from an incremental backup."""
        trades_path = os.path.join(backup_path, "trades.json")
        if os.path.exists(trades_path):
            with open(trades_path, "r") as f:
                trades = json.load(f)
            if trades.get("ids"):
                self.vector_store.trade_collection.upsert(
                    ids=trades["ids"],
                    embeddings=trades["embeddings"],
                    metadatas=trades["metadatas"],
                    documents=trades["documents"],
                )
        patterns_path = os.path.join(backup_path, "patterns.json")
        if os.path.exists(patterns_path):
            with open(patterns_path, "r") as f:
                patterns = json.load(f)
            if patterns.get("ids"):
                self.vector_store.pattern_collection.upsert(
                    ids=patterns["ids"],
                    embeddings=patterns["embeddings"],
                    metadatas=patterns["metadatas"],
                    documents=patterns["documents"],
                )
        return True

    def validate_integrity(self) -> bool:
        """Validate data integrity of the vector store."""
        try:
            self.vector_store.get_trade_count()
            self.vector_store.get_pattern_count()
            return True
        except Exception:
            return False
