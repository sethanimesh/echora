"""Create local transcription records.

Revision ID: 0001_local_transcription_records
Revises:
Create Date: 2026-08-13
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0001_local_transcription_records"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transcriptions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("language", sa.String(length=16), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("audio_storage_name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("audio_storage_name"),
    )
    op.create_table(
        "asr_hypotheses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("transcription_id", sa.String(length=36), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("score_type", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["transcription_id"], ["transcriptions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_asr_hypotheses_transcription_id", "asr_hypotheses", ["transcription_id"])
    op.create_table(
        "asr_segments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("hypothesis_id", sa.Integer(), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["hypothesis_id"], ["asr_hypotheses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_asr_segments_hypothesis_id", "asr_segments", ["hypothesis_id"])


def downgrade() -> None:
    op.drop_index("ix_asr_segments_hypothesis_id", table_name="asr_segments")
    op.drop_table("asr_segments")
    op.drop_index("ix_asr_hypotheses_transcription_id", table_name="asr_hypotheses")
    op.drop_table("asr_hypotheses")
    op.drop_table("transcriptions")
