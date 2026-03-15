from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from base.base_model import CustomBaseModel as BaseModel


class BaseService(ABC, BaseModel):
    """Abstract base class for synchronous services.

    Implementations must provide a `process` method that accepts
    an input payload and returns a result. Inheriting from
    `CustomBaseModel` allows services to include Pydantic fields
    if desired.
    """

    @abstractmethod
    def process(self, inputs: Any) -> Any:
        """Process inputs synchronously.

        Args:
            inputs: The input data for the service. Type is intentionally
                broad to support diverse use cases.

        Returns:
            The processed result. Concrete services should document the
            specific return type.
        """
        raise NotImplementedError()


class AsyncBaseService(ABC, BaseModel):
    """Abstract base class for asynchronous services.

    Implementations must provide an async `process` method.
    """

    @abstractmethod
    async def process(self, inputs: Any) -> Any:
        """Process inputs asynchronously.

        Args:
            inputs: The input data for the service.

        Returns:
            The processed result.
        """
        raise NotImplementedError()
