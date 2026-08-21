import pytest
from ops import OP_BY_NAME, OP_NAMES, OPS

# (op_name, input, expected) — correctness cases for every op, plus edge cases.
CASES = [
    ("SORT", [4, 1, 3], [1, 3, 4]),
    ("SORT", [5], [5]),
    ("SORT", [9, 8, 7, 6, 5], [5, 6, 7, 8, 9]),
    ("RUNMAX", [6, 9, 5, 6], [6, 9, 9, 9]),
    ("RUNMAX", [3], [3]),
    ("RUNMIN", [6, 9, 5, 6], [6, 6, 5, 5]),
    ("DEDUP", [4, 0, 0, 1, 0], [4, 0, 1]),
    ("DEDUP", [3, 3, 3], [3]),
    ("UNION", [9, 2, 8, 2], [2, 8, 9]),
    ("ROTATE", [1, 2, 3], [3, 1, 2]),
    ("ROTATE", [5], [5]),
    ("REVERSE", [1, 2, 3], [3, 2, 1]),
    ("PARITY", [9, 3, 6], [0]),
    ("PARITY", [9, 3, 6, 9], [1]),
    ("PARITY", [2, 4, 6], [0]),
    ("SORT_DESC", [4, 1, 3], [4, 3, 1]),
    ("SORT_DESC", [5], [5]),
    ("MAX", [4, 1, 3], [4]),
    ("MAX", [7], [7]),
    ("MIN", [4, 1, 3], [1]),
    ("MODE", [4, 1, 4, 3, 1, 1], [1]),
    ("MODE", [4, 1, 3], [1]),
    ("MODE", [7, 7], [7]),
    ("ROTATE_LEFT", [1, 2, 3], [2, 3, 1]),
    ("ROTATE_LEFT", [5], [5]),
    ("EVENS_IDX", [5, 6, 7, 8, 9], [5, 7, 9]),
    ("EVENS_IDX", [9], [9]),
    ("ODDS_IDX", [5, 6, 7, 8, 9], [6, 8]),
    ("ODDS_IDX", [9], []),
    ("FILTER_EVEN", [4, 1, 3, 2], [4, 2]),
    ("FILTER_EVEN", [1, 3], []),
    ("FILTER_ODD", [4, 1, 3, 2], [1, 3]),
    ("REMOVE_MAX", [4, 1, 4, 3], [1, 3]),
    ("REMOVE_MAX", [5], []),
    ("REMOVE_MIN", [4, 1, 3, 1], [4, 3]),
    ("FIRST", [4, 1, 3], [4]),
    ("LAST", [4, 1, 3], [3]),
]

VALID_FAMILIES = {"compare", "set", "move", "reduce", "filter"}


@pytest.mark.parametrize("name, xs, want", CASES)
def test_op_correct(name, xs, want):
    assert OP_BY_NAME[name].solve(xs) == want


@pytest.mark.parametrize("name, xs, want", CASES)
def test_op_no_mutation(name, xs, want):
    original = xs[:]
    OP_BY_NAME[name].solve(xs)
    assert xs == original


def test_op_count():
    assert len(OPS) == 21


def test_names_unique():
    assert len(set(OP_NAMES)) == len(OP_NAMES)


def test_op_by_name_consistent():
    for op in OPS:
        assert OP_BY_NAME[op.name] is op


def test_families_valid():
    for op in OPS:
        assert op.family in VALID_FAMILIES


def test_every_op_has_a_case():
    # no registered op may go untested
    tested = {name for name, _, _ in CASES}
    assert tested == set(OP_NAMES)
