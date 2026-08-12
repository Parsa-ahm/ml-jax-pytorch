import pytest
from ops import (
    OP_BY_NAME,
    OP_NAMES,
    OPS,
    Op,
    dedup,
    parity,
    reverse,
    rotate,
    runmax,
    runmin,
    sort,
    union,
)


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 3], [1, 3, 4]),
        ([5], [5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([9, 8, 7, 6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_sort(xs, want):
    original = xs[:]
    assert sort(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 3], [4, 4, 4]),
        ([5], [5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 9, 8, 7, 6, 5, 4, 3, 2, 1], [1, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
    ],
)
def test_runmax(xs, want):
    original = xs[:]
    assert runmax(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 3], [4, 1, 1]),
        ([5], [5]),
        ([1, 2, 3, 4, 5], [1, 1, 1, 1, 1]),
    ],
)
def test_runmin(xs, want):
    original = xs[:]
    assert runmin(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 1], [4, 1]),
        ([5], [5]),
        ([1, 1, 3, 3, 5], [1, 3, 5]),
    ],
)
def test_dedup(xs, want):
    original = xs[:]
    assert dedup(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 3], [1, 3, 4]),
        ([5], [5]),
        ([5, 2, 1, 2, 3, 4, 5, 2, 4], [1, 2, 3, 4, 5]),
    ],
)
def test_union(xs, want):
    original = xs[:]
    assert union(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 3], [3, 4, 1]),
        ([5], [5]),
        ([1, 2, 3, 4, 5], [5, 1, 2, 3, 4]),
    ],
)
def test_rotate(xs, want):
    original = xs[:]
    assert rotate(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 3], [3, 1, 4]),
        ([5], [5]),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
    ],
)
def test_reverse(xs, want):
    original = xs[:]
    assert reverse(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "xs, want",
    [
        ([4, 1, 3], [0]),
        ([5], [1]),
        ([1, 2, 3, 4, 5], [1]),
    ],
)
def test_parity(xs, want):
    original = xs[:]
    assert parity(xs) == want
    assert xs == original


@pytest.mark.parametrize(
    "item, want",
    [
        (OPS[0], Op("SORT", "compare", solve=sort)),
        (OPS[1], Op("RUNMAX", "compare", solve=runmax)),
        (OPS[2], Op("RUNMIN", "compare", solve=runmin)),
        (OPS[3], Op("DEDUP", "set", solve=dedup)),
        (OPS[4], Op("UNION", "set", solve=union)),
        (OPS[5], Op("ROTATE", "move", solve=rotate)),
        (OPS[6], Op("REVERSE", "move", solve=reverse)),
        (OPS[7], Op("PARITY", "reduce", solve=parity)),
    ],
)
def test_ops(item, want):
    assert item == want


@pytest.mark.parametrize(
    "item, want",
    [
        (OP_NAMES[0], "SORT"),
        (OP_NAMES[1], "RUNMAX"),
        (OP_NAMES[2], "RUNMIN"),
        (OP_NAMES[3], "DEDUP"),
        (OP_NAMES[4], "UNION"),
        (OP_NAMES[5], "ROTATE"),
        (OP_NAMES[6], "REVERSE"),
        (OP_NAMES[7], "PARITY"),
    ],
)
def test_op_name(item, want):
    assert item == want


@pytest.mark.parametrize(
    "item, want",
    [
        (OP_BY_NAME["SORT"], Op("SORT", "compare", solve=sort)),
        (OP_BY_NAME["RUNMAX"], Op("RUNMAX", "compare", solve=runmax)),
        (OP_BY_NAME["RUNMIN"], Op("RUNMIN", "compare", solve=runmin)),
        (OP_BY_NAME["DEDUP"], Op("DEDUP", "set", solve=dedup)),
        (OP_BY_NAME["UNION"], Op("UNION", "set", solve=union)),
        (OP_BY_NAME["ROTATE"], Op("ROTATE", "move", solve=rotate)),
        (OP_BY_NAME["REVERSE"], Op("REVERSE", "move", solve=reverse)),
        (OP_BY_NAME["PARITY"], Op("PARITY", "reduce", solve=parity)),
    ],
)
def test_op_by_name(item, want):
    assert item == want
