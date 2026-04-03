# =============================================================================
# models.py — Modelli ORM (traduce da python a sql) e Pydantic schemas
# =============================================================================

import uuid
from datetime import datetime
from typing import Optional, List, Literal
from sqlalchemy import String, Boolean, Text, ForeignKey, JSON, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from pydantic import BaseModel, EmailStr, Field
from app.database import Base


# =============================================================================
# SEZIONE 1 — ORM MODELS
# =============================================================================

class User(Base):
    """
    Tabella utenti — credenziali e profilo.
    hashed_password: la password non viene mai salvata in chiaro.
    google_calendar_id: nullable, serve solo se l'utente usa il Calendar Agent.
    """
    __tablename__ = "users"

    id               : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email            : Mapped[str]       = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password  : Mapped[str]       = mapped_column(String(255), nullable=False)
    full_name        : Mapped[str]       = mapped_column(String(255), nullable=False)
    is_active        : Mapped[bool]      = mapped_column(Boolean, default=True)
    created_at       : Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())
    google_calendar_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relazione 1→N: un utente ha molte conversazioni
    conversations: Mapped[List["Conversation"]] = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """
    Tabella conversazioni — ogni sessione di chat è una Conversation.
    summary: campo TEXT che persiste il riassunto cumulativo della history.
             Viene aggiornato da summarize_node ogni summary_threshold messaggi.
             Inizia vuoto e cresce progressivamente — mai perso.
    """
    __tablename__ = "conversations"

    id         : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id    : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title      : Mapped[str]       = mapped_column(String(500), nullable=False)
    summary    : Mapped[str]       = mapped_column(Text, default="")   # summary memory persistita
    created_at : Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at : Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relazioni
    user     : Mapped["User"]          = relationship("User", back_populates="conversations")
    messages : Mapped[List["Message"]] = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """
    Tabella messaggi — ogni singolo turno di conversazione.
    role: "human" (input utente) o "ai" (risposta del sistema).
    agente_usato: quale sotto-agente ha risposto (hr_agent, ml_agent, ecc.).
    tools_usati: lista dei tool chiamati durante la risposta, salvata come JSON.
    """
    __tablename__ = "messages"

    id              : Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id : Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role            : Mapped[str]            = mapped_column(String(10), nullable=False)   # "human" | "ai"
    content         : Mapped[str]            = mapped_column(Text, nullable=False)
    agente_usato    : Mapped[Optional[str]]  = mapped_column(String(50), nullable=True)
    tools_usati     : Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at      : Mapped[datetime]       = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relazione N→1
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")


# =============================================================================
# SEZIONE 2 — PYDANTIC SCHEMAS
# =============================================================================
# Convenzione di naming:
#   *Create  → dati in entrata per creare una risorsa (request body)
#   *Response → dati in uscita verso il client (response body)

# --- Auth ---

class UserCreate(BaseModel):
    """Body di POST /auth/register"""
    email     : EmailStr
    password  : str = Field(min_length=8)
    full_name : str = Field(min_length=2)

class UserResponse(BaseModel):
    """Response di register e get_current_user — MAI esporre hashed_password"""
    id         : uuid.UUID
    email      : str
    full_name  : str
    is_active  : bool
    created_at : datetime

    model_config = {"from_attributes": True}  # permette di costruire da ORM object

class TokenResponse(BaseModel):
    """Response di POST /auth/login e /auth/refresh"""
    access_token  : str
    refresh_token : str
    token_type    : str = "bearer"

# --- Chat ---

class ChatRequest(BaseModel):
    """Body di POST /v1/chat"""
    query           : str
    conversation_id : Optional[uuid.UUID] = None  # None = nuova conversazione

class ChatResponse(BaseModel):
    """Response di POST /v1/chat"""
    risposta        : str
    agente_usato    : str
    conversation_id : uuid.UUID
    tools_usati     : List[str] = []

# --- Supervisor routing ---

class RoutingDecision(BaseModel):
    """
    Output strutturato del Supervisor.
    Usato con with_structured_output(RoutingDecision) — MAI JsonOutputParser.
    agente: quale sotto-agente invocare.
    """
    agente            : Literal["hr_agent", "ml_agent", "report_agent", "calendar_agent"]
    motivazione       : str   # perché il supervisor ha scelto questo agente
    query_riformulata : str   # query ottimizzata per il sotto-agente

# --- Calendar HITL ---

class CalendarEventRequest(BaseModel):
    """Dettagli evento da approvare — restituito in pending_approval"""
    titolo      : str
    data        : str   # formato YYYY-MM-DD
    ora_inizio  : str   # formato HH:MM
    ora_fine    : str   # formato HH:MM
    descrizione : str = ""

class ApproveRequest(BaseModel):
    """Body di POST /v1/approve"""
    conversation_id : uuid.UUID
    approved        : bool
