from __future__ import annotations
from dataclasses import dataclass
from contextvars import ContextVar

from recsys.db.repositories.ratings import RatingsRepository
from recsys.db.repositories.movies import MoviesRepository

@dataclass
class RequestContext:
    ratings: RatingsRepository
    movies: MoviesRepository

request_ctx_var: ContextVar[RequestContext | None] = ContextVar("request_ctx", default=None)

def set_request_ctx(ctx: RequestContext) -> None:
    request_ctx_var.set(ctx)

def get_request_ctx() -> RequestContext:
    ctx = request_ctx_var.get()
    if ctx is None:
        raise RuntimeError("RequestContext is not set. Did you forget the dependency?")
    return ctx