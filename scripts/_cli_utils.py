"""Shared argparse type validators for scripts."""

from __future__ import annotations

import argparse
import math


def positive_int(value: str) -> int:
    """argparse type: integer >= 1."""
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer: {value!r}") from exc
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1: {n}")
    return n


def nonneg_int(value: str) -> int:
    """argparse type: integer >= 0."""
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer: {value!r}") from exc
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0: {n}")
    return n


def positive_float(value: str) -> float:
    """argparse type: float > 0."""
    try:
        n = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number: {value!r}") from exc
    if not math.isfinite(n):
        raise argparse.ArgumentTypeError(f"must be finite: {value!r}")
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0: {n}")
    return n


def nonneg_float(value: str) -> float:
    """argparse type: float >= 0."""
    try:
        n = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number: {value!r}") from exc
    if not math.isfinite(n):
        raise argparse.ArgumentTypeError(f"must be finite: {value!r}")
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0: {n}")
    return n


def finite_float(value: str) -> float:
    """argparse type: finite float."""
    try:
        n = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a number: {value!r}") from exc
    if not math.isfinite(n):
        raise argparse.ArgumentTypeError(f"must be finite: {value!r}")
    return n
