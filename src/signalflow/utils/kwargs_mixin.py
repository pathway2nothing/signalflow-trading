from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable, TypeVar
from loguru import logger

T = TypeVar("T")


class KwargsTolerantMixin:
    """
    Mixin for dataclasses:
    - accepts extra **kwargs in __init__
    - ignores them (doesn't store)
    - logs what was ignored + what fields exist
    """

    __ignore_unknown_kwargs__: bool = True
    __log_unknown_kwargs__: bool = True
    __strict_unknown_kwargs__: bool = False

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        orig_init = cls.__dict__.get("__init__")
        if orig_init is None:
            return

        def wrapped_init(self: T, *args: Any, **kwargs2: Any) -> None:
            if not kwargs2:
                return orig_init(self, *args)

            try:
                known = {f.name for f in fields(self)}
            except TypeError:
                return orig_init(self, *args, **kwargs2)

            unknown = {k: v for k, v in kwargs2.items() if k not in known}
            passed = {k: v for k, v in kwargs2.items() if k in known}

            if unknown:
                if getattr(cls, "__strict_unknown_kwargs__", False):
                    raise TypeError(
                        f"{cls.__name__} got unexpected kwargs: {sorted(unknown)}. Available: {sorted(known)}"
                    )

                if getattr(cls, "__log_unknown_kwargs__", True):
                    logger.warning(
                        "Ignored kwargs for {}: {}",
                        cls.__name__,
                        sorted(unknown),
                    )
                    logger.info(
                        "Available fields for {}: {}",
                        cls.__name__,
                        sorted(known),
                    )

                if not getattr(cls, "__ignore_unknown_kwargs__", True):
                    passed.update(unknown)

            return orig_init(self, *args, **passed)

        wrapped_init.__name__ = orig_init.__name__
        wrapped_init.__qualname__ = orig_init.__qualname__
        wrapped_init.__doc__ = orig_init.__doc__

        setattr(cls, "__init__", wrapped_init)
