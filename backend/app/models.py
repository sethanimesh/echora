"""Persistence models for locally retained transcription attempts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for persisted audit records."""

    return datetime.now(UTC)


class Transcription(Base):
    """One uploaded audio file and its normalized ASR response."""

    __tablename__ = "transcriptions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="en")
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    audio_storage_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )

    hypotheses: Mapped[list[ASRHypothesis]] = relationship(
        back_populates="transcription",
        cascade="all, delete-orphan",
        order_by="ASRHypothesis.rank",
    )


class ASRHypothesis(Base):
    """A raw text hypothesis returned by an ASR provider."""

    __tablename__ = "asr_hypotheses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transcription_id: Mapped[str] = mapped_column(
        ForeignKey("transcriptions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    score_type: Mapped[str] = mapped_column(String(64), nullable=False)

    transcription: Mapped[Transcription] = relationship(back_populates="hypotheses")
    segments: Mapped[list[ASRSegment]] = relationship(
        back_populates="hypothesis",
        cascade="all, delete-orphan",
        order_by="ASRSegment.segment_index",
    )


class ASRSegment(Base):
    """An optional timestamped segment belonging to one raw hypothesis."""

    __tablename__ = "asr_segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hypothesis_id: Mapped[int] = mapped_column(
        ForeignKey("asr_hypotheses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)

    hypothesis: Mapped[ASRHypothesis] = relationship(back_populates="segments")
