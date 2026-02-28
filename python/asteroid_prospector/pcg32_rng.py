"""Deterministic PCG32 RNG used by parity-critical paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MASK64 = (1 << 64) - 1
_MASK32 = (1 << 32) - 1


@dataclass
class Pcg32Rng:
    """PCG32 RNG with helper distributions mirroring native core usage."""

    seed: int
    stream: int = 54

    def __post_init__(self) -> None:
        self._state = 0
        self._inc = ((int(self.stream) & _MASK64) << 1) | 1
        self._next_u32()
        self._state = (self._state + (int(self.seed) & _MASK64)) & _MASK64
        self._next_u32()

    def _next_u32(self) -> int:
        oldstate = self._state
        xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) & _MASK32
        rot = (oldstate >> 59) & 31

        self._state = (oldstate * 6364136223846793005 + self._inc) & _MASK64
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & _MASK32

    def _next_f64(self) -> float:
        return float(self._next_u32() / 4294967296.0)

    def _draw_exponential_unit(self) -> float:
        u = self._next_f64()
        if u < 1.0e-8:
            u = 1.0e-8
        return float(-np.log(u))

    def _draw_normal(self, mean: float, sigma: float) -> float:
        u1 = self._next_f64()
        u2 = self._next_f64()
        if u1 < 1.0e-8:
            u1 = 1.0e-8
        mag = float(np.sqrt(-2.0 * np.log(u1)))
        z0 = mag * float(np.cos(2.0 * np.pi * u2))
        return float(mean + sigma * z0)

    def _sample_scalar(self, draw_fn, size: tuple[int, ...]) -> np.ndarray:
        arr = np.empty(size, dtype=np.float64)
        flat = arr.reshape(-1)
        for i in range(flat.size):
            flat[i] = draw_fn()
        return arr

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> int | np.ndarray:
        lo = int(low)
        hi = int(high) if high is not None else lo
        if high is None:
            lo = 0

        span = hi - lo
        if span <= 0:
            raise ValueError("high must be greater than low")

        def draw() -> int:
            return lo + (self._next_u32() % span)

        if size is None:
            return int(draw())

        shape = (size,) if isinstance(size, int) else tuple(size)
        arr = np.empty(shape, dtype=np.int64)
        flat = arr.reshape(-1)
        for i in range(flat.size):
            flat[i] = draw()
        return arr

    def random(self, size: int | tuple[int, ...] | None = None) -> float | np.ndarray:
        if size is None:
            return self._next_f64()

        shape = (size,) if isinstance(size, int) else tuple(size)
        return self._sample_scalar(self._next_f64, shape)

    def uniform(
        self,
        low: float | np.ndarray = 0.0,
        high: float | np.ndarray = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        lo_arr = np.asarray(low, dtype=np.float64)
        hi_arr = np.asarray(high, dtype=np.float64)

        if size is None and lo_arr.ndim == 0 and hi_arr.ndim == 0:
            return float(lo_arr) + (float(hi_arr) - float(lo_arr)) * self._next_f64()

        if size is None:
            target_shape = np.broadcast_shapes(lo_arr.shape, hi_arr.shape)
        else:
            target_shape = (size,) if isinstance(size, int) else tuple(size)

        lo_b = np.broadcast_to(lo_arr, target_shape)
        hi_b = np.broadcast_to(hi_arr, target_shape)
        out = np.empty(target_shape, dtype=np.float64)
        flat_out = out.reshape(-1)
        flat_lo = lo_b.reshape(-1)
        flat_hi = hi_b.reshape(-1)

        for i in range(flat_out.size):
            flat_out[i] = flat_lo[i] + (flat_hi[i] - flat_lo[i]) * self._next_f64()

        return out

    def normal(
        self,
        loc: float | np.ndarray = 0.0,
        scale: float | np.ndarray = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        loc_arr = np.asarray(loc, dtype=np.float64)
        scale_arr = np.asarray(scale, dtype=np.float64)

        if size is None and loc_arr.ndim == 0 and scale_arr.ndim == 0:
            return self._draw_normal(float(loc_arr), float(scale_arr))

        if size is None:
            target_shape = np.broadcast_shapes(loc_arr.shape, scale_arr.shape)
        else:
            target_shape = (size,) if isinstance(size, int) else tuple(size)

        loc_b = np.broadcast_to(loc_arr, target_shape)
        scale_b = np.broadcast_to(scale_arr, target_shape)
        out = np.empty(target_shape, dtype=np.float64)
        flat_out = out.reshape(-1)
        flat_loc = loc_b.reshape(-1)
        flat_scale = scale_b.reshape(-1)

        for i in range(flat_out.size):
            flat_out[i] = self._draw_normal(float(flat_loc[i]), float(flat_scale[i]))

        return out

    def lognormal(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        vals = self.normal(mean, sigma, size)
        return float(np.exp(vals)) if np.isscalar(vals) else np.exp(vals, dtype=np.float64)

    def _gamma(self, shape: float) -> float:
        if shape <= 0.0:
            raise ValueError("shape must be positive")

        rounded = int(round(shape))
        if abs(shape - float(rounded)) < 1.0e-12 and rounded > 0:
            total = 0.0
            for _ in range(rounded):
                total += self._draw_exponential_unit()
            return total

        # Marsaglia and Tsang method for non-integer shapes.
        if shape < 1.0:
            u = self._next_f64()
            return self._gamma(shape + 1.0) * (u ** (1.0 / shape))

        d = shape - 1.0 / 3.0
        c = 1.0 / np.sqrt(9.0 * d)
        while True:
            x = self._draw_normal(0.0, 1.0)
            v = (1.0 + c * x) ** 3
            if v <= 0.0:
                continue
            u = self._next_f64()
            if u < 1.0 - 0.0331 * (x**4):
                return d * v
            if np.log(u) < 0.5 * x * x + d * (1.0 - v + np.log(v)):
                return d * v

    def beta(
        self,
        a: float,
        b: float,
        size: int | tuple[int, ...] | None = None,
    ) -> float | np.ndarray:
        def draw() -> float:
            ga = self._gamma(float(a))
            gb = self._gamma(float(b))
            total = ga + gb
            return 0.5 if total <= 0.0 else ga / total

        if size is None:
            return float(draw())

        shape = (size,) if isinstance(size, int) else tuple(size)
        return self._sample_scalar(draw, shape)

    def dirichlet(
        self,
        alpha: np.ndarray,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray:
        alpha_arr = np.asarray(alpha, dtype=np.float64)
        if alpha_arr.ndim != 1:
            raise ValueError("alpha must be a 1D vector")

        def one_draw() -> np.ndarray:
            vals = np.empty(alpha_arr.shape, dtype=np.float64)
            for i, a in enumerate(alpha_arr):
                vals[i] = self._gamma(float(a))
            total = float(np.sum(vals))
            if total <= 0.0:
                vals.fill(1.0 / float(vals.size))
                return vals
            return vals / total

        if size is None:
            return one_draw()

        shape = (size,) if isinstance(size, int) else tuple(size)
        out = np.empty(shape + alpha_arr.shape, dtype=np.float64)
        flat = out.reshape(-1, alpha_arr.size)
        for i in range(flat.shape[0]):
            flat[i, :] = one_draw()
        return out
