"""Contains Implementations of sorting and searching algorithms"""

from typing import List, TypeVar

T = TypeVar("T", int, float)


def bubble_sort(array: List[T]) -> None:
    """Performs bubble sort on array
    T(n) = O(n^2)
    Space Complexity: O(1)

    Parameters
    ----------
    array : List[T]
        elements that need to be sorted
    """
    n = len(array)
    for i in range(n):
        swapped = False
        for j in range(n - i + 1):
            if array[j + 1] < array[j]:
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True
        if not swapped:
            break


def recursive_bubble_sort(array: List[T]) -> None:
    """performs bubble sort recursively
    T(n) = O(n^2)
    Space Complexity: O(1)

    Parameters
    ----------
    array : List[T]
        array to be sorted
    """
    for idx, num in enumerate(array):
        try:
            if array[idx + 1] < num:
                array[idx] = array[idx + 1]
                array[idx + 1] = num
                recursive_bubble_sort(array)
        except IndexError:
            continue


def selection_sort(array: List[T]) -> None:
    pass
