from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from base import AsyncBaseService

from .models import Base
from .settings import PGSettings


class PostgresClient(AsyncBaseService):
    """PostgreSQL database client service."""

    settings: PGSettings

    @cached_property
    def sessionmaker(self) -> sessionmaker[Session]:
        """Create and return a SQLAlchemy `sessionmaker` bound to the DB engine.

        This method lazily constructs the engine and creates all ORM tables.
        """
        engine = create_engine(
            f"postgresql+psycopg2://{self.settings.username}:{self.settings.password}@{self.settings.host}/{self.settings.db}",
        )
        Base.metadata.create_all(engine)
        return sessionmaker(autoflush=False, bind=engine)

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Context manager that yields a SQLAlchemy `Session`.

        Ensures the session is closed after use.
        """
        session = None
        try:
            session = self.sessionmaker()
            yield session
        finally:
            if session:
                session.close()

    async def check_health(self) -> bool:
        """Asynchronously check database connectivity by acquiring a session.

        Returns:
            True when a session can be created (DB reachable), otherwise False.
        """
        try:
            with self.get_session():
                return True
        except Exception as e:
            # In a real-world app, you'd log the exception before returning/raising.
            raise e

    async def process(self, inputs) -> bool:
        """
        Minimal async `process` implementation to satisfy `AsyncBaseService`.

        For now this simply proxies to `check_health`, making it suitable for
        generic health-check style usage.
        """
        return await self.check_health()
