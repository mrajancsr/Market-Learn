# pyre-strict
"""Helper tools for making life easier

Author: Rajan Subramanian
Date: 10/20/2020
"""


from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar

# pyre-ignore
T = TypeVar("T", bound=Callable[[Any], Any])


def timethis(func: T) -> T:
    """Decorator that reports execution time

    Parameters
    ----------
    func : T
        Callable function that you want to decorate

    Returns
    -------
    T
        called function with execution time
    """

    @wraps(func)
    # pyre-ignore
    def wrapper(*args, **kwargs) -> Any:
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"Finished {func.__name__!r} in {(end-start):.4f} secs")
        return result

    return wrapper
