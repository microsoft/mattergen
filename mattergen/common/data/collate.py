# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar, overload


from torch import Tensor
from torch_geometric.data import Batch, Data
from typing_extensions import TypeGuard

warnings.filterwarnings(
    "ignore", "TypedStorage is deprecated", module="torch_geometric"
)  # Till https://github.com/pyg-team/pytorch_geometric/pull/7034 is released.

__all__ = ["collate", "find_structure", "separate"]

TreeTypes = Data | Batch | Tensor | int | float | str | bool | None
T = TypeVar("T", bound=TreeTypes)
PyTree = T | list["PyTree[T]"] | tuple["PyTree[T]", ...] | dict[Any, "PyTree[T]"]
IterPyTree = list[PyTree[T]] | tuple[PyTree[T], ...] | dict[Any, PyTree[T]]


@overload
def collate(x: PyTree[T]) -> T:
    ...


@overload
def collate(x: PyTree[T], depth: int | None) -> PyTree[T]:
    ...


def collate(x: PyTree[T], depth: int | None = None) -> T | PyTree[T]:
    """Collate over the `depth` outermost layers of a `PyTree[Data]`, where `depth = None` collates
    over the whole structure.

    The type `PyTree[T]` is defined recursively, in the following way::

        PyTree[T] = Union[T, list[PyTree[T]], tuple[PyTree[T], ...], dict[Any, PyTree[T]]]]

    The following are examples of a `PyTree[int]`::

        1
        [1, 2]
        [1, (2, 3)]
        [(1, 2), (3, 4)]
        [{"key": [1, 2]}, 3, (4, 5)]
        [{"key": [1, 2]}, {"key": [3, 4]}, {"key": [5, 6]}]

    If every `Union` in a `PyTree` consists of one and only one element, then the `PyTree` is
    called _consistent_. A consistent `PyTree[T]` can be decomposed into layers, where the first
    layer is also referred to as the outermost layer and the last layer is referred to as the
    innermost layer. This decomposition into layers is also defined recursively.

    If `list[U]` is a consistent `PyTree`, then the outermost/first layer of `list[U]` is `list`,
    and the `n`th layer of `list[U]` is the `n - 1`th layer of `U`. Similarly, if `tuple[U, ...]` is
    a consistent `PyTree`, then the outermost/first layer is `tuple` and the `n`th layer is the
    `n - 1`th layer of `U`. Finally, if `dict[Any, U]` is a consistent `PyTree`, then the
    outermost/first layer is `dict` and the `n`th layer is the `n - 1`th layer of `U`.

    For the above examples of a `PyTree[int]`::

        # Consistent, but has no layers:
        1

        # Consistent with one layer `list`:
        [1, 2]

        # Inconsistent:
        [1, (2, 3)]

        # Consistent with outermost layer `list` and innermost layer `tuple`:
        [(1, 2), (3, 4)]

        # Inconsistent:
        [{"key": [1, 2]}, 3, (4, 5)]

        # Consistent with outermost layer `list`, second layer `dict`, and innermost layer `list`:
        [{"key": [1, 2]}, {"key": [3, 4]}, {"key": [5, 6]}]

    A few examples of how `collate` would work for various values of `depth`::

        # Collate over everything:
        collate(x: list[dict[str, tuple[Data, Data]]]) -> Data

        # Collate only over the outermost `list`:
        collate(x: list[dict[str, tuple[Data, Data]]], depth=1) -> dict[str, tuple[Data, Data]]

        # Collate over the outermost layer `list` and the second layer `dict`:
        collate(x: list[dict[str, tuple[Data, Data]]], depth=2) -> tuple[Data, Data]

    The inverse function of :func:`collate` is :func:`separate`.

    Args:
        x (PyTree[Data]): The data structure to collate.
        depth (int, optional): Number of outermost layers to collate over. If given, `x` must be
            a consistent `PyTree`. If not given, `x` needs not to be consistent, and this function
            will collate over the whole structure.

    Raises:
        ValueError: If `x` is not a `PyTree`. Also raised if `depth` is specified but `x` is not a
            consistent `PyTree`.

    Returns:
        PyTree[T]: Collated structure.
    """
    ys, structure, _ = _flatten(x, depth, 0)
    return _merge(ys, structure)


def _flatten_iterable(
    xs: Iterable[PyTree[T]],
    depth: int | None,
    offset: int,
) -> tuple[list[PyTree[T]], list[PyTree[int]], int]:
    ys, ss = [], []
    for x in xs:
        y, s, offset = _flatten(x, depth, offset)
        ys.append(y)
        ss.append(s)
    return sum(ys, []), ss, offset


def iter_leaves(x: PyTree[T]) -> Iterator[T]:
    """Iterate over the leaves of a `DataTree`.

    Args:
        x (PyTree[T]): The data structure to iterate over.

    Yields:
        T: The leaves of `x`.
    """
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from iter_leaves(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from iter_leaves(y)
    else:
        yield x


def len_tree(x: PyTree[T]) -> int:
    """Number of nodes in a `PyTree`.

    Args:
        x (PyTree[T]): The data structure to iterate over.

    Returns:
        int: Number of nodes in `x`.
    """
    total = 0
    if isinstance(x, (list, tuple)):
        for y in x:
            total += len_tree(y)
    elif isinstance(x, dict):
        for y in x.values():
            total += len_tree(y)
    else:
        total = 1
    return total


def _flatten(
    xs: PyTree[T],
    depth: int | None = None,
    offset: int = 0,
) -> tuple[list[PyTree[T]], PyTree[int], int]:
    depth = None if depth is None else depth - 1

    if isinstance(xs, Data) or depth == -1:
        return [xs], offset, offset + 1

    if isinstance(xs, list):
        ys, ss, offset = _flatten_iterable(xs, depth, offset)
        return ys, list(ss), offset

    if isinstance(xs, tuple):
        ys, ss, offset = _flatten_iterable(xs, depth, offset)
        return ys, tuple(ss), offset

    if isinstance(xs, dict):
        keys = sorted(xs.keys())
        ys, ss, offset = _flatten_iterable([xs[k] for k in keys], depth, offset)
        return ys, {k: s for k, s in zip(keys, ss)}, offset

    raise ValueError(f"Cannot flatten item of type `{type(xs)}`.")


def is_list_seq(xs: Sequence[PyTree[T]]) -> TypeGuard[Sequence[list[PyTree[T]]]]:
    """Check if a sequence of `PyTree`s is a sequence of lists of `PyTree`s."""
    return all(isinstance(x, list) for x in xs)


def is_data_seq(xs: Sequence[PyTree[T]]) -> TypeGuard[Sequence[Data]]:
    """Check if a sequence of `PyTree`s is a sequence of Data objects."""
    return all(isinstance(x, Data) for x in xs)


def is_tuple_seq(xs: Sequence[PyTree[T]]) -> TypeGuard[Sequence[tuple[PyTree[T]]]]:
    """Check if a sequence of `PyTree`s is a sequence of `tuple`s of `PyTree`s."""
    return all(isinstance(x, tuple) for x in xs)


def is_dict_seq(xs: Sequence[PyTree[T]]) -> TypeGuard[Sequence[dict[Any, PyTree[T]]]]:
    """Check if a sequence of `PyTree`s is a sequence of `dict`s with `PyTree` values."""
    return all(isinstance(x, dict) for x in xs)


def _merge(xs: list[PyTree[T]], structure: PyTree[int]) -> PyTree[T]:
    if len(xs) == 0:
        raise ValueError("Cannot merge a sequence of length zero.")

    # Check for consistency.
    types = set(type(x) for x in xs)
    if len(types) != 1:
        raise ValueError(f"`PyTree` is inconsistent. Found a mix of {len(types)} types: `{types}`.")

    if is_data_seq(xs):
        # Intersection of attrs:
        attrs = set(
            xs[0].keys() if callable(xs[0].keys) else xs[0].keys
        )  # pyg < 2.4.0 compatibility
        for x in xs[1:]:
            attrs.intersection_update(
                x.keys() if callable(x.keys) else x.keys
            )  # pyg < 2.4.0 compatibility

        # Filter attrs that are not in the intersection:
        for x in xs:
            for attr in list(x.keys() if callable(x.keys) else x.keys):  # pyg < 2.4.0 compatibility
                if attr not in attrs:
                    warnings.warn(
                        f"Attribute `{attr}` is not in the intersection of attributes of "
                        f"the collated `Data` objects. This attribute will be dropped."
                    )
                    del x[attr]  # type: ignore

        try:
            batch = Batch.from_data_list(xs)
        except Exception as e:
            # Check if dtypes do not match:
            for attr in attrs:
                # Check types:
                types = set(type(x[attr]) for x in xs)
                if len(types) != 1:
                    raise ValueError(
                        f"Attribute `{attr}` has inconsistent types. Found a mix of "
                        f"{len(types)} types: `{types}`."
                    )

                # Check dtypes
                if isinstance(xs[0][attr], Tensor):
                    dtypes = set(x[attr].dtype for x in xs)
                    if len(dtypes) != 1:
                        raise ValueError(
                            f"Attribute `{attr}` has inconsistent dtypes. Found a mix of "
                            f"{len(dtypes)} dtypes: `{dtypes}`."
                        )

            raise e
        # Save the structure information as a hidden attribute. This is also what
        # :func:`Batch.from_data_list` does.
        batch._collate_structure = structure
        return batch

    if is_list_seq(xs):
        return [_merge(list(ys), structure) for ys in zip(*xs)]

    if is_tuple_seq(xs):
        return tuple(_merge(list(ys), structure) for ys in zip(*xs))

    if is_dict_seq(xs):
        return {k: _merge([x[k] for x in xs], structure) for k in xs[0].keys()}

    raise ValueError(f"Cannot merge elements of type `{type(xs[0])}`.")


def separate(
    x: PyTree[T],
    structure: PyTree[int] | None = None,
) -> PyTree[T]:
    """Inverse of :func:`collate`. This function guarantees that the following is true for every
    value of `depth`::

        separate(collate(x, depth)) == x

    Args:
        x (PyTree[Data] or PyTree[Tensor]): Data structure which is structured like the output of
            :func:`collate`.
        structure (PyTree[int], optional): If `x` is a `PyTree[Data]`, then this argument can
            be ignored (usually). If `x` is a `PyTree[Tensor]`, then :func:`separate` needs to be
            told how the result should be separated into the original `PyTree`. In this case, you
            should run :func:`find_structure` on the output of :func:`collate` and pass the result
            as this argument.

    Raises:
        RuntimeError: If :func:`separate` cannot automatically infer how to separate `x`.
        ValueError: If `x` is not a `PyTree[Data]` or `PyTree[Tensor]`.

    Returns:
        PyTree[Data] or PyTree[Tensor]: `x` separated into the `PyTree` originally given to
            :func:`collate`.
    """
    if structure is None:
        structure = find_structure(x)
    return _separate(x, structure)


def tree_map(
    func: Callable[..., T],
    x: PyTree[T],
    *x2: PyTree[T],
) -> PyTree[T]:
    """Apply `func` to every leaf in `x`.

     Args:
        x (PyTree[T]): `PyTree`s to map over.
        *x2 (PyTree[Any]): additional matching `PyTree`s possibly of different type to map over.
        func (function): Function to apply.

    Returns:
        PyTree[T]: `x`, but with `func` applied to every leaf.
    """

    # Nested function to prevent recursively defining of generic T.
    def _map(x: PyTree[T], *x2: PyTree) -> PyTree[T]:
        if isinstance(x, list):
            assert is_list_seq(x2), "All `PyTree`s must of the same form, but they are not."
            return [_map(*y) for y in zip(x, *x2)]
        elif isinstance(x, tuple):
            assert is_tuple_seq(x2), "All `PyTree`s must of the same form, but they are not."
            return tuple(_map(*y) for y in zip(x, *x2))
        elif isinstance(x, dict):
            assert is_dict_seq(x2), "All `PyTree`s must of the same form, but they are not."
            # Check if all keys match
            if any(
                any(k[0] != k2_k for k2_k in k[1:])
                for k in zip(x.keys(), *map(lambda a: a.keys(), x2))
            ):
                raise ValueError("Cannot merge dictionaries with different keys.")
            return {
                y[0]: _map(y[1], *y[2:])
                for y in zip(x.keys(), x.values(), *map(lambda a: a.values(), x2))
            }
        else:
            return func(x, *x2)

    return _map(x, *x2)


def find_structure(x: PyTree[T]) -> IterPyTree[int]:
    """Find the information necessary to structure something back into the original `PyTree` given
    to :func:`collate`. The output of this function can be given as the second argument to
    :func:`separate`.

    Args:
        x (PyTree[Data] or PyTree[Tensor]): Collated data structure. This is usually the output of
            :func:`collate`.

    Raises:
        RuntimeError: If `x` does not contain the necessary structure information.

    Returns:
        PyTree[int]: Structure information.
    """
    if isinstance(x, Data):
        if not hasattr(x, "_collate_structure"):
            raise RuntimeError(
                "The attribute `_collate_structure` is necessary to separate the collated batch, "
                "but this attribute cannot be found. It might have been lost along the way. "
                "You can use `find_structure` to extract the structure information directly from "
                "the output of `collate` and then pass this to `separate` as the second argument."
            )
        return x._collate_structure

    if isinstance(x, (list, tuple)):
        return find_structure(x[0])

    if isinstance(x, dict):
        return find_structure(list(x.values())[0])

    raise RuntimeError(
        "The structure information necessary to separate the collated batch is not contained in "
        "the input. "
        "You can use `find_structure` to extract the structure information directly from "
        "the output of `collate` and then pass this to `separate` as the second argument."
    )


def _separate(x: PyTree[T], structure: PyTree[int]) -> PyTree[T]:
    if isinstance(structure, int):
        return _get_i(x, structure)

    if isinstance(structure, list):
        return [_separate(x, s) for s in structure]

    if isinstance(structure, tuple):
        return tuple(_separate(x, s) for s in structure)

    if isinstance(structure, dict):
        return {k: _separate(x, v) for k, v in structure.items()}

    raise ValueError(f"Cannot reconstruct object of type `{type(structure)}`.")


def _get_i(xs: PyTree[T], i: int) -> PyTree[T]:
    if isinstance(xs, Data):
        return xs.get_example(i)

    if isinstance(xs, Tensor):
        return xs[i]  # type: ignore  # mypy does not understand that this is a Tensor.

    if isinstance(xs, list):
        return list(_get_i(x, i) for x in xs)

    if isinstance(xs, tuple):
        return tuple(_get_i(x, i) for x in xs)

    if isinstance(xs, dict):
        return {k: _get_i(v, i) for k, v in xs.items()}

    raise ValueError(f"Cannot get example for `{type(xs)}`.")
