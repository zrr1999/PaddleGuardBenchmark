from __future__ import annotations

import paddle

from src.utils import benchmark_fn


def add(a: paddle.Tensor, b: paddle.Tensor) -> paddle.Tensor:
    return a + b


def test_simple(benchmark):
    a = paddle.randn([10, 10])
    b = paddle.randn([10, 10])
    benchmark_fn(add, benchmark)(a, b)
