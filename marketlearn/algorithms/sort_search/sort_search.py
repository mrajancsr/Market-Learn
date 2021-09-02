"""Contains Implementations of sorting and searching algorithms"""

from typing import List


def bubble_sort(array: List[float]) -> None:
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


def recursive_bubble_sort(array: List[float]) -> None:
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


def selection_sort(array: List[float]) -> None:
    n = len(array)
    for i in range(n):
        # assume first index is minimum
        min_idx = i
        for j in range(i + 1, n):
            if array[min_idx] < array[j]:
                min_idx = j
        array[i], array[min_idx] = array[min_idx], array[i]


def insertion_sort(array: List[float]) -> None:
    n = len(array)
    for i in range(1, n):
        key = array[i]
        # move elements in array[0,..i-1] that are greater than key
        # to one position ahead of current position
        j = i - 1
        while j >= 0 and key < array[j]:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key
