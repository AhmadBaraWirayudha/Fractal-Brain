"""
fractal_brain/signal.py
Signal processing: retardation (delay lines) and 1D convolution.
"""
from .math_utils import Vector

class DelayLine:
    """A FIFO buffer that holds past values (of any type -- scalars, Vectors, ...)."""
    def __init__(self, max_delay):
        self.max_delay = max_delay
        # None means "nothing pushed here yet" -- deliberately not 0.0, since callers may
        # push Vectors, and a stray float placeholder would silently masquerade as a real
        # (falsy-but-present) value of the wrong type until the buffer had fully cycled.
        self.buffer = [None] * max_delay
        self.pos = 0

    def push(self, value):
        delayed = self.buffer[self.pos]
        self.buffer[self.pos] = value
        self.pos = (self.pos + 1) % self.max_delay
        return delayed   # the value that just left, or None if this slot was never filled

    def tap(self, delay):
        """Read a value delayed by `delay` steps (0 = current, 1 = one step ago, ...)."""
        idx = (self.pos - 1 - delay) % self.max_delay
        return self.buffer[idx]


def convolve1d(signal, kernel):
    """
    Perform 1D valid convolution of signal (list or Vector) with kernel (list or Vector).
    Returns list of output values.
    """
    if isinstance(signal, Vector):
        signal = signal.data
    if isinstance(kernel, Vector):
        kernel = kernel.data
    out_len = len(signal) - len(kernel) + 1
    out = []
    for i in range(out_len):
        val = sum(signal[i+j] * kernel[j] for j in range(len(kernel)))
        out.append(val)
    return out