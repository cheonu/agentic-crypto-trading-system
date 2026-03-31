import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from agentic_crypto_trading_system.memory.vector_store import VectorStore
from agentic_crypto_trading_system.memory.memory_service import MemoryService
from agentic_crypto_trading_system.memory.backup import MemoryBackupManager


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def vector_store(tmp_dir):
    """Create a VectorStore with a temp directory."""
    return VectorStore(persist_directory=os.path.join(tmp_dir, "chromadb"))


@pytest.fixture
def mock_embedding():
    """Return a fake 384-dim embedding (MiniLM size)."""
    return np.random.rand(384).tolist()


def test_vector_store_add_and_query_trade(vector_store, mock_embedding):
    """Test adding and querying a trade memory."""
    vector_store.add_trade(
        trade_id="trade-1",
        embedding=mock_embedding,
        metadata={"symbol": "BTC/USDT", "regime": "bull"},
        document="long BTC/USDT trade during bull regime",
    )
    assert vector_store.get_trade_count() == 1

    results = vector_store.query_trades(query_embedding=mock_embedding, n_results=1)
    assert results["ids"][0][0] == "trade-1"


def test_vector_store_add_and_query_pattern(vector_store, mock_embedding):
    """Test adding and querying a pattern memory."""
    vector_store.add_pattern(
        pattern_id="pattern-1",
        embedding=mock_embedding,
        metadata={"symbol": "ETH/USDT", "regime": "bear"},
        document="breakout pattern on ETH/USDT",
    )
    assert vector_store.get_pattern_count() == 1

    results = vector_store.query_patterns(query_embedding=mock_embedding, n_results=1)
    assert results["ids"][0][0] == "pattern-1"


def test_vector_store_delete(vector_store, mock_embedding):
    """Test deleting memories."""
    vector_store.add_trade("t1", mock_embedding, {"symbol": "BTC/USDT"}, "doc")
    assert vector_store.get_trade_count() == 1
    vector_store.delete_trade("t1")
    assert vector_store.get_trade_count() == 0


def test_vector_store_upsert(vector_store, mock_embedding):
    """Test that upsert overwrites existing entries."""
    vector_store.add_trade("t1", mock_embedding, {"symbol": "BTC/USDT"}, "original")
    vector_store.add_trade("t1", mock_embedding, {"symbol": "BTC/USDT"}, "updated")
    assert vector_store.get_trade_count() == 1


@patch("agentic_crypto_trading_system.memory.memory_service.SentenceTransformer")
def test_memory_service_store_and_query_trade(mock_st_class, vector_store):
    """Test MemoryService store and query trade with mocked embeddings."""
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: np.random.rand(384).tolist())
    mock_st_class.return_value = mock_model

    service = MemoryService(vector_store=vector_store)
    service.store_trade_outcome(
        trade_id="trade-1",
        symbol="BTC/USDT",
        direction="long",
        entry_price=50000.0,
        exit_price=51000.0,
        pnl=100.0,
        pnl_pct=2.0,
        regime="bull",
        agent_id="agent-1",
    )
    assert vector_store.get_trade_count() == 1

    results = service.query_similar_trades("BTC long trade in bull market")
    assert len(results["ids"][0]) == 1


@patch("agentic_crypto_trading_system.memory.memory_service.SentenceTransformer")
def test_memory_service_store_and_query_pattern(mock_st_class, vector_store):
    """Test MemoryService store and query pattern with mocked embeddings."""
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: np.random.rand(384).tolist())
    mock_st_class.return_value = mock_model

    service = MemoryService(vector_store=vector_store)
    service.store_pattern(
        pattern_id="p1",
        description="Double bottom reversal",
        pattern_type="reversal",
        symbol="ETH/USDT",
        regime="bear",
        confidence=0.85,
        agent_id="agent-2",
    )
    assert vector_store.get_pattern_count() == 1

    results = service.query_patterns("reversal pattern", regime="bear")
    assert len(results["ids"][0]) == 1


def test_backup_incremental_and_restore(vector_store, mock_embedding, tmp_dir):
    """Test incremental backup and restore."""
    backup_dir = os.path.join(tmp_dir, "backups")
    manager = MemoryBackupManager(vector_store, backup_directory=backup_dir)

    vector_store.add_trade("t1", mock_embedding, {"symbol": "BTC/USDT"}, "trade doc")
    vector_store.add_pattern("p1", mock_embedding, {"symbol": "ETH/USDT"}, "pattern doc")

    backup_path = manager.create_incremental_backup()
    assert os.path.exists(os.path.join(backup_path, "manifest.json"))
    assert os.path.exists(os.path.join(backup_path, "trades.json"))
    assert os.path.exists(os.path.join(backup_path, "patterns.json"))


def test_validate_integrity(vector_store, tmp_dir):
    """Test integrity validation."""
    manager = MemoryBackupManager(vector_store, backup_directory=tmp_dir)
    assert manager.validate_integrity() is True
