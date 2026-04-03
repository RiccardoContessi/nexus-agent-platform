from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator
from app.config import get_settings

settings = get_settings()

# -----------------------------------------------------------------------------
# Engine — connessione al DB, singleton per tutta l'app
# -----------------------------------------------------------------------------
async_engine = create_async_engine(
    settings.postgres_url,
    echo=False,
    pool_pre_ping=True,   # verifica che la connessione sia viva prima di usarla
    pool_size=10,         # connessioni mantenute aperte nel pool
    max_overflow=20,      # connessioni extra consentite sotto carico
)

# -----------------------------------------------------------------------------
# AsyncSession factory — produce sessioni su richiesta
# -----------------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
# sessione = AsyncSessionLocal()               # ORA crei una sessione reale


# -----------------------------------------------------------------------------
# Base — classe padre di tutti i modelli ORM (in models.py)
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass

# -----------------------------------------------------------------------------
# get_db() — dependency FastAPI
# -----------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()    # commit automatico se tutto ok
        except Exception:
            await session.rollback()  # rollback automatico in caso di errore
            raise

