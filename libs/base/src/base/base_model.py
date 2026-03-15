from __future__ import annotations

from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    """Pydantic BaseModel with project-specific defaults.

    The `Config` subclass sets `arbitrary_types_allowed` to True so
    that models can contain non-Pydantic types when necessary.
    """

    class Config:
        """Configuration of the Pydantic Object"""

        # Allowing arbitrary types for class validation
        arbitrary_types_allowed = True
