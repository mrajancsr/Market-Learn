"""Helper tools for makeing life easier"""

import time
from functools import wraps
from typing import Callable


def timethis(func: Callable):
    """Decorator that reports execution time

    :param func: the function you want to decorate
    :type func: Callable
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Finished {func.__name__!r} in {(end-start):.4f} secs")
        return result
    return wrapper


def debugthis(func: Callable):
    """Decorator that prints function signature and return values

    :param func: the function you want to decorate
    :type func: Callable
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # create a list of positional arguments
        args_repr = [repr(arg) for arg in args]

        # create a list of keyword arguments, '!r' means repr
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]

        # join the positional and keyword arguments
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper

