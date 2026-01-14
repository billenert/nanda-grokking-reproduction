"""Mathematical utilities."""

import math


def is_prime(n: int) -> bool:
    """Check if a number is prime.

    Args:
        n: Number to check.

    Returns:
        True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def next_prime(n: int) -> int:
    """Find the next prime number >= n.

    Args:
        n: Starting number.

    Returns:
        The smallest prime number >= n.
    """
    while not is_prime(n):
        n += 1
    return n
