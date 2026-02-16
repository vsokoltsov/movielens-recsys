from __future__ import annotations
from dataclasses import dataclass
from contextvars import ContextVar

from recsys.repository.protocols import RatingsRepositoryProtocol
from recsys.repository.protocols import MoviesRepositoryProtocol


@dataclass
class RequestContext:
    ratings: RatingsRepositoryProtocol
    movies: MoviesRepositoryProtocol


request_ctx_var: ContextVar[RequestContext | None] = ContextVar(
    "request_ctx", default=None
)


def set_request_ctx(ctx: RequestContext) -> None:
    request_ctx_var.set(ctx)


def get_request_ctx() -> RequestContext:
    ctx = request_ctx_var.get()
    if ctx is None:
        raise RuntimeError("RequestContext is not set. Did you forget the dependency?")
    return ctx
