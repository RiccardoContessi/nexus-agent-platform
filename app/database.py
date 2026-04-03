# =============================================================================
# database.py — Infrastruttura di connessione a PostgreSQL
# =============================================================================
# Questo file non sa nulla del dominio (utenti, conversazioni, messaggi).
# Sa solo UNA cosa: come aprire e chiudere connessioni al database.
#
# Stack:
#   - SQLAlchemy 2.x    → ORM + query builder
#   - asyncpg           → driver PostgreSQL non-bloccante (sotto il cofano)
#   - AsyncSession      → sessione asincrona per ogni richiesta HTTP
#
# Flusso di vita di una sessione:
#   FastAPI riceve richiesta
#     → get_db() apre AsyncSession
#     → endpoint esegue query
#     → get_db() chiude sessione (anche in caso di eccezione)
# =============================================================================

# Modelli ORM
# ORM (Object-Relational Mapper) è un layer che traduce tra 
# il mondo Python e il mondo SQL. Invece di scrivere INSERT INTO users VALUES (...) 
# scrivi db.add(User(email="...", full_name="...")) e SQLAlchemy genera il SQL 
# corretto per te. I "modelli ORM" sono le classi Python che rappresentano le 
# tabelle — class User(Base) rappresenta la tabella users. 
# Ogni istanza di quella classe è una riga della tabella.
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
# create_async_engine usa automaticamente asyncpg come driver
# perché la URL inizia con "postgresql+asyncpg://"
# echo=False in produzione — metti True per vedere le query SQL nei log
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
# expire_on_commit=False: dopo un commit, gli oggetti ORM rimangono
# accessibili senza dover fare un'altra query al DB
# di base: SQLAlchemy svuota oggetti ORM intenzionalmente perché pensa: "il DB potrebbe aver cambiato qualcosa durante il commit, meglio ricaricare".
# configuri come le sessioni verranno create
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
# sessione = AsyncSessionLocal()               # ORA crei una sessione reale


# -----------------------------------------------------------------------------
# Base — classe padre di tutti i modelli ORM (in models.py)
# -----------------------------------------------------------------------------
# Tutti i modelli ereditano da Base — SQLAlchemy usa questa gerarchia
# per sapere quali classi mappare sulle tabelle del DB
class Base(DeclarativeBase):
    pass

# -----------------------------------------------------------------------------
# get_db() — dependency FastAPI
# -----------------------------------------------------------------------------
# Ogni endpoint che ha bisogno del DB dichiara:
#   async def my_endpoint(db: AsyncSession = Depends(get_db))
#
# FastAPI chiama get_db(), ottiene la sessione, la passa all'endpoint.
# Il finally garantisce che la sessione venga sempre chiusa,
# anche se l'endpoint lancia un'eccezione.
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()    # commit automatico se tutto ok
        except Exception:
            await session.rollback()  # rollback automatico in caso di errore
            raise


# =============================================================================
# COSA FA QUESTO FILE IN UNA RIGA:
# Configura COME l'app parla con PostgreSQL. Non tocca mai i dati.
# =============================================================================

# OGGETTI CREATI (usati nel resto del progetto):
#
# async_engine       → la connessione fisica al DB. Singleton, vive per tutta l'app.
#
# AsyncSessionLocal  → factory che produce sessioni. Non è una sessione,
#                      è lo stampo per crearle. Ogni richiesta HTTP ne crea una nuova.
#
# Base               → classe padre dei modelli ORM in models.py.
#                      Serve a SQLAlchemy per sapere quali classi sono tabelle.
#
# get_db()           → dependency FastAPI. Apre una sessione, la passa all'endpoint,
#                      committa o fa rollback, la chiude. Automatico, non ci pensi.

# FLUSSO DI UNA RICHIESTA HTTP:
# 1. FastAPI chiama get_db()
# 2. get_db() chiama AsyncSessionLocal() → crea sessione
# 3. sessione viene passata all'endpoint tramite Depends(get_db)
# 4. endpoint esegue le query
# 5. get_db() committa (o rollback se errore) e chiude la sessione