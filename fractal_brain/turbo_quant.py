"""
fractal_brain/turbo_quant.py
8‑bit integer quantization / dequantization for matrices and vectors.
Uses affine (min-max) linear quantization: value ≈ quantized_byte * scale + min_val.
No external dependencies.
"""
from .math_utils import Matrix, Vector

class TurboQuant:
    """
    Compress a Matrix or Vector to 8‑bit integer representation (range [0,255]).
    Only the raw bytes are stored; we provide methods to quantize/dequantize.
    """
    @staticmethod
    def quantize_matrix(mat: Matrix):
        """Return (quantized_bytes, scale, min_val, shape) for a Matrix.
        `shape` is required by dequantize_matrix to reconstruct the row/col layout."""
        flat = []
        for row in mat.data:
            flat.extend(row)
        min_val = min(flat)
        max_val = max(flat)
        if max_val == min_val:
            scale = 1.0
        else:
            scale = (max_val - min_val) / 255.0
        bytes_ = []
        for val in flat:
            q = int(round((val - min_val) / scale))
            q = max(0, min(255, q))
            bytes_.append(q)
        return bytes_, scale, min_val, mat.shape

    @staticmethod
    def dequantize_matrix(quantized_bytes, scale, min_val, shape):
        """Reconstruct a Matrix from quantized data."""
        rows, cols = shape
        flat = [(b * scale) + min_val for b in quantized_bytes]
        data = []
        idx = 0
        for _ in range(rows):
            data.append(flat[idx:idx+cols])
            idx += cols
        return Matrix(data)

    @staticmethod
    def quantize_vector(vec: Vector):
        """Return (quantized_bytes, scale, min_val) for a Vector."""
        flat = vec.data
        min_val = min(flat)
        max_val = max(flat)
        if max_val == min_val:
            scale = 1.0
        else:
            scale = (max_val - min_val) / 255.0
        bytes_ = []
        for val in flat:
            q = int(round((val - min_val) / scale))
            q = max(0, min(255, q))
            bytes_.append(q)
        return bytes_, scale, min_val

    @staticmethod
    def dequantize_vector(quantized_bytes, scale, min_val):
        """Reconstruct a Vector from quantized data."""
        return Vector([(b * scale) + min_val for b in quantized_bytes])