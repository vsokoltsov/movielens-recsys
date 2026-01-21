from __future__ import annotations

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    BigInteger,
    CheckConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    gender = Column(String(1), nullable=True)  # 'M'/'F'
    age = Column(Integer, nullable=True)
    occupation = Column(Integer, nullable=True)
    zip = Column(String(16), nullable=True)


class Movie(Base):
    __tablename__ = "movies"

    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    genres = Column(Text, nullable=True)


class Rating(Base):
    __tablename__ = "ratings"

    # Если в датасете нет уникального id, логично сделать составной PK:
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    movie_id = Column(Integer, ForeignKey("movies.movie_id", ondelete="CASCADE"), primary_key=True)

    rating = Column(Integer, nullable=False)
    timestamp = Column(BigInteger, nullable=True)

    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_ratings_rating_range"),
        Index("ix_ratings_user_id", "user_id"),
        Index("ix_ratings_movie_id", "movie_id"),
    )