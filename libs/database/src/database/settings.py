from __future__ import annotations

from base import BaseModel


class PGSettings(BaseModel):
    """Database connection settings.

    Attributes:
        username: Database username.
        password: Database password.
        host: Database host address or DSN.
        db: Database name.
    """

    username: str
    password: str
    host: str
    db: str
