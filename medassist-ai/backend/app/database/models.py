import enum
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, ForeignKey, DateTime, Enum
)
from sqlalchemy.orm import relationship

from app.database.database import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    MEDICAL_STUDENT = "medical_student"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.MEDICAL_STUDENT)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="uploader")
    conversations = relationship("Conversation", back_populates="user")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), nullable=True)  # e.g. Cardiology, Diabetes
    file_path = Column(String(500), nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"))
    embedding_status = Column(String(50), default="pending")  # pending|processing|complete|failed
    created_at = Column(DateTime, default=datetime.utcnow)

    uploader = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    page_number = Column(Integer, nullable=True)
    section = Column(String(255), nullable=True)
    chunk_text = Column(Text, nullable=False)
    embedding_id = Column(Integer, nullable=True)  # position in FAISS index

    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    sources = relationship("RetrievedSource", back_populates="conversation", cascade="all, delete-orphan")


class RetrievedSource(Base):
    __tablename__ = "retrieved_sources"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    chunk_id = Column(Integer, ForeignKey("chunks.id"))
    similarity_score = Column(Float, nullable=False)

    conversation = relationship("Conversation", back_populates="sources")
