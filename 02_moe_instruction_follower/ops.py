"""The 21 list operations and the Op registry (name, family, solve)."""

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


def sort_dec(xs: list[int]) -> list[int]:
    ans = xs[:]
    ans.sort(reverse=True)
    return ans


def max_num(xs: list[int]) -> list[int]:
    return [max(xs)]


def min_num(xs: list[int]) -> list[int]:
    return [min(xs)]


def first_num(xs: list[int]) -> list[int]:
    return [xs[0]]


def last_num(xs: list[int]) -> list[int]:
    return [xs[-1]]


def mode(xs: list[int]) -> list[int]:
    counts = {}
    for i in xs:
        counts[i] = counts.get(i, 0) + 1
    best = max(counts.values())
    return [min(x for x in counts if counts[x] == best)]


def rotate_left(xs: list[int]) -> list[int]:
    return xs[1:] + xs[:1]


def even_index(xs: list[int]) -> list[int]:
    return xs[0::2]


def odd_index(xs: list[int]) -> list[int]:
    return xs[1::2]


def filter_even(xs: list[int]) -> list[int]:
    ans = []
    for i in xs:
        if i % 2 == 0:
            ans.append(i)
    return ans


def filter_odd(xs: list[int]) -> list[int]:
    ans = []
    for i in xs:
        if i % 2 == 1:
            ans.append(i)
    return ans


def remove_max(xs: list[int]) -> list[int]:
    m = max(xs)
    return [x for x in xs if x != m]


def remove_min(xs: list[int]) -> list[int]:
    m = min(xs)
    return [x for x in xs if x != m]


Family = Literal["compare", "set", "move", "reduce", "filter"]


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
SortDesc = Op("SORT_DESC", "compare", solve=sort_dec)
Max = Op("MAX", "compare", solve=max_num)
Min = Op("MIN", "compare", solve=min_num)
Mode = Op("MODE", "set", solve=mode)
RotateLeft = Op("ROTATE_LEFT", "move", solve=rotate_left)
EvensIdx = Op("EVENS_IDX", "move", solve=even_index)
OddsIdx = Op("ODDS_IDX", "move", solve=odd_index)
FilterEven = Op("FILTER_EVEN", "filter", solve=filter_even)
FilterOdd = Op("FILTER_ODD", "filter", solve=filter_odd)
RemoveMax = Op("REMOVE_MAX", "filter", solve=remove_max)
RemoveMin = Op("REMOVE_MIN", "filter", solve=remove_min)
First = Op("FIRST", "reduce", solve=first_num)
Last = Op("LAST", "reduce", solve=last_num)

OPS = (
    Sort,
    Runmax,
    Runmin,
    Dedup,
    Union,
    Rotate,
    Reverse,
    Parity,
    SortDesc,
    Max,
    Min,
    Mode,
    RotateLeft,
    EvensIdx,
    OddsIdx,
    FilterEven,
    FilterOdd,
    RemoveMax,
    RemoveMin,
    First,
    Last,
)

OP_NAMES = tuple(op.name for op in OPS)

OP_BY_NAME = {op.name: op for op in OPS}
