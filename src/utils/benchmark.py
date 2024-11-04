from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import paddle
from paddle.jit.sot import symbolic_translate


def benchmark_fn[**P](
    fn: Callable[P, paddle.Tensor],
    benchmark: Any,
    *,
    repeat: int = 1000,
    warmup: int = 10,
):
    sym_fn = symbolic_translate(fn)

    def wrapper(*inputs: P.args, **kwargs: P.kwargs):
        output = fn(*inputs, **kwargs)
        sym_output = sym_fn(*inputs, **kwargs)
        np.testing.assert_allclose(sym_output, output)
        for _ in range(warmup):
            sym_fn(*inputs, **kwargs)

        @benchmark
        def benchmark_fn():
            for _ in range(repeat):
                sym_fn(*inputs, **kwargs)

    return wrapper
