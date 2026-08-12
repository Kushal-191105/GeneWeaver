"""
DNA sequence numeric encoding and decoding component for GPU processing.
Translates ACGTN DNA sequences into 8-bit unsigned integer arrays.

Encoding specification:
    A / a -> 0
    C / c -> 1
    G / g -> 2
    T / t -> 3
    N / n -> 4 (ambiguous base)
"""

from typing import Union, Sequence
import numpy as np

# Encoding mapping definitions
BASE_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}

INT_TO_BASE = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
    4: "N",
}

# 256-entry lookup table for ultra-fast ASCII byte to uint8 mapping
_ENCODE_LUT = np.full(256, 255, dtype=np.uint8)
_ENCODE_LUT[ord("A")] = 0
_ENCODE_LUT[ord("a")] = 0
_ENCODE_LUT[ord("C")] = 1
_ENCODE_LUT[ord("c")] = 1
_ENCODE_LUT[ord("G")] = 2
_ENCODE_LUT[ord("g")] = 2
_ENCODE_LUT[ord("T")] = 3
_ENCODE_LUT[ord("t")] = 3
_ENCODE_LUT[ord("N")] = 4
_ENCODE_LUT[ord("n")] = 4

# Decoding lookup table
_DECODE_LUT = np.array([ord("A"), ord("C"), ord("G"), ord("T"), ord("N")], dtype=np.uint8)

# Validation base sets aligned with existing project rules
VALID_BASES = set("ATGC")
AMBIGUOUS_BASES = set("N")
ALLOWED_BASES = set("ATGCNatgcn")


def validate_dna_sequence(sequence: str) -> bool:
    """
    Validate sequence characters against project rules (A, C, G, T, N).

    Args:
        sequence: DNA string to validate.

    Returns:
        bool: True if all characters are valid, False otherwise.
    """
    if not sequence:
        return True
    return set(sequence.upper()).issubset(set("ATGCN"))


def encode_sequence(sequence: str, validate: bool = True) -> np.ndarray:
    """
    Encode a DNA string into a NumPy uint8 array for CUDA kernel consumption.

    Args:
        sequence: DNA string (case-insensitive).
        validate: If True, raises ValueError when unknown base characters are encountered.

    Returns:
        np.ndarray: 1D array of dtype uint8 with values in [0..4].

    Raises:
        ValueError: If invalid DNA characters are found and validate=True.
    """
    if not isinstance(sequence, str):
        sequence = str(sequence)

    if len(sequence) == 0:
        return np.empty(0, dtype=np.uint8)

    # Convert string to ASCII byte array
    ascii_bytes = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
    encoded = _ENCODE_LUT[ascii_bytes]

    if validate:
        invalid_indices = np.where(encoded == 255)[0]
        if len(invalid_indices) > 0:
            first_invalid_pos = invalid_indices[0]
            invalid_char = sequence[first_invalid_pos]
            raise ValueError(
                f"Invalid DNA base '{invalid_char}' at position {first_invalid_pos}. "
                f"Expected one of 'A', 'C', 'G', 'T', 'N'."
            )

    return encoded


def decode_sequence(encoded: Union[np.ndarray, Sequence[int]]) -> str:
    """
    Decode a numeric uint8 array back to a DNA string.

    Args:
        encoded: 1D array or sequence of integers in [0..4].

    Returns:
        str: Decoded DNA string (e.g. 'ACGTN').

    Raises:
        ValueError: If integer values outside [0..4] are present.
    """
    arr = np.asarray(encoded, dtype=np.uint8)
    if arr.size == 0:
        return ""

    if np.any(arr > 4):
        invalid_val = arr[np.where(arr > 4)[0][0]]
        raise ValueError(f"Cannot decode numeric value {invalid_val} to DNA base. Valid range is 0-4.")

    decoded_bytes = _DECODE_LUT[arr]
    return decoded_bytes.tobytes().decode("ascii")


def encode_target(target: str) -> np.ndarray:
    """
    Encode a CRISPR target sequence into a uint8 array.

    Args:
        target: Target DNA sequence.

    Returns:
        np.ndarray: Encoded target array of uint8.
    """
    return encode_sequence(target.strip(), validate=True)
