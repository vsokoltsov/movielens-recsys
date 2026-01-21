"""seed movielens ml-1m data

Revision ID: 987c10513f96
Revises: de48eeb5924c
Create Date: 2026-01-19 18:35:46.689025

"""
from typing import Sequence, Union

import os
from alembic import op
import sqlalchemy as sa
from recsys.config import RAW_BUCKET
from recsys.gcp import GCPStorageClient
from recsys.utils import read_from_csv

# revision identifiers, used by Alembic.
revision: str = '987c10513f96'
down_revision: Union[str, Sequence[str], None] = 'de48eeb5924c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    client = GCPStorageClient()
    if RAW_BUCKET is not None:
        client.download(
            bucket_name=RAW_BUCKET,
            object_name="ml-1m/users.dat",
            dst_path=os.path.join("/tmp", "users.dat")
        )
        client.download(
            bucket_name=RAW_BUCKET,
            object_name="ml-1m/movies.dat",
            dst_path=os.path.join("/tmp", "movies.dat")
        )
        client.download(
            bucket_name=RAW_BUCKET,
            object_name="ml-1m/ratings.dat",
            dst_path=os.path.join("/tmp", "ratings.dat")
        )
        users_df = read_from_csv(
            path=os.path.join("/tmp", "users.dat"),
            columns=["user_id", "gender", "age", "occupation", "zip"]
        )
        ratings_df = read_from_csv(
            path=os.path.join("/tmp", "ratings.dat"),
            columns=["user_id", "movie_id", "rating", "timestamp"]
        )
        movies_df = read_from_csv(
            path=os.path.join("/tmp", "movies.dat"),
            columns=["movie_id", "title", "genres"]
        )
        users_df.to_sql(
            name="users",
            con=conn,
            if_exists="append",
            index=False
        )
        movies_df.to_sql(
            name="movies",
            con=conn,
            if_exists="append",
            index=False
        )
        ratings_df.to_sql(
            name="ratings",
            con=conn,
            if_exists="append",
            index=False
        )


def downgrade() -> None:
    """Downgrade schema."""
    pass
