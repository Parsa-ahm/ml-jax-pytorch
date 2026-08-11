"""
Ops.py hold the deterministic operations for the 8 functions we will be training on:
0 -> sort
1 -> running max
2 -> running min
3 -> de duplicate
4 -> union set
5 -> rotate
6 -> reverse
7 -> parity (the mod 2 of the number of odd numbers in a list)

these functions are written in a purely functional manner where no in place mutations or
side effects happen.

The Operations are in a Op data class tuple at the bottom which includes their
name, family, and function

Accessible through the:
OPS tuple of Op
OP_Names tuple of names
OP_BY_NAME dictionary {name : Op}
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


def sort(xs: list[int]) -> list[int]:
    result = xs[:]
    result.sort()
    return result


def runmax(xs: list[int]) -> list[int]:
    result = xs[:]
    m = result[0]
    for i in range(len(result)):
        if result[i] > m:
            m = result[i]
        result[i] = m
    return result


def runmin(xs: list[int]) -> list[int]:
    result = xs[:]
    m = result[0]
    for i in range(len(result)):
        if result[i] < m:
            m = result[i]
        result[i] = m
    return result


def dedup(xs: list[int]) -> list[int]:
    seen = set()
    ans = []
    for i in xs:
        if i not in seen:
            seen.add(i)
            ans.append(i)
    return ans


def union(xs: list[int]) -> list[int]:
    result = dedup(xs)
    result = sort(result)
    return result


def rotate(xs: list[int]) -> list[int]:
    return xs[-1:] + xs[:-1]


def reverse(xs: list[int]) -> list[int]:
    return xs[::-1]


def parity(xs: list[int]) -> list[int]:
    count = 0
    for i in xs:
        if i % 2 == 1:
            count += 1
    return [count % 2]


Family = Literal["compare", "set", "move", "reduce"]


@dataclass(frozen=True)
class Op:
    name: str
    family: Family
    solve: Callable[[list[int]], list[int]]


Sort = Op("SORT", "compare", solve=sort)
Runmax = Op("RUNMAX", "compare", solve=runmax)
Runmin = Op("RUNMIN", "compare", solve=runmin)
Dedup = Op("DEDUP", "set", solve=dedup)
Union = Op("UNION", "set", solve=union)
Rotate = Op("ROTATE", "move", solve=rotate)
Reverse = Op("REVERSE", "move", solve=reverse)
Parity = Op("PARITY", "reduce", solve=parity)

OPS = (Sort, Runmax, Runmin, Dedup, Union, Rotate, Reverse, Parity)

OP_NAMES = tuple(op.name for op in OPS)

OP_BY_NAME = {op.name: op for op in OPS}
