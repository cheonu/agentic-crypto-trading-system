from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

class DatabaseConnection:
    """Manages database connections with pooling."""

    def __init__(self, database_url: str, pool_size: int = 5, max_overflow: int = 10):
        # Async engine
        self.async_engine = create_async_engine(
            database_url.replace ("postgresql://", "postgresql+asyncpg://"),
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=False,   
        ) 
        self.async_session_factory = async_sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )     

        # Sync engine (for migrations)
        sync_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        self.sync_engine = create_engine(sync_url, pool_size=pool_size)
        self.sync_session_factory = sessionmaker(bind=self.sync_engine)

    async def create_tables(self):
        """Create all tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self):
        """Drop all tables."""
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def get_async_session(self):
        """Provide an async session."""
        return self.async_session_factory() 

    async def close(self):
        """Close all connections."""
        await self.async_engine_dispose()


