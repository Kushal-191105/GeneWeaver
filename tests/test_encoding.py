"""
Tests for DNA numeric encoding and decoding.
"""

import pytest
import numpy as np
from src.gpu.encoding import (
    encode_sequence,
    decode_sequence,
    encode_target,
    validate_dna_sequence,
    BASE_TO_INT,
    INT_TO_BASE,
)


def test_base_mappings():
    assert BASE_TO_INT["A"] == 0
    assert BASE_TO_INT["C"] == 1
    assert BASE_TO_INT["G"] == 2
    assert BASE_TO_INT["T"] == 3
    assert BASE_TO_INT["N"] == 4

    assert INT_TO_BASE[0] == "A"
    assert INT_TO_BASE[1] == "C"
    assert INT_TO_BASE[2] == "G"
    assert INT_TO_BASE[3] == "T"
    assert INT_TO_BASE[4] == "N"


def test_encode_sequence_standard():
    seq = "ACGTN"
    encoded = encode_sequence(seq)
    assert isinstance(encoded, np.ndarray)
    assert encoded.dtype == np.uint8
    np.testing.assert_array_equal(encoded, np.array([0, 1, 2, 3, 4], dtype=np.uint8))


def test_encode_sequence_case_insensitive():
    seq = "acgtnACGTN"
    encoded = encode_sequence(seq)
    expected = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.uint8)
    np.testing.assert_array_equal(encoded, expected)


def test_encode_sequence_empty():
    encoded = encode_sequence("")
    assert encoded.size == 0
    assert encoded.dtype == np.uint8


def test_encode_sequence_invalid_character_raises():
    with pytest.raises(ValueError, match="Invalid DNA base 'X'"):
        encode_sequence("ACGTXACGT")

    with pytest.raises(ValueError, match="Invalid DNA base '9'"):
        encode_sequence("ACGT9")


def test_decode_sequence():
    arr = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0], dtype=np.uint8)
    decoded = decode_sequence(arr)
    assert decoded == "ACGTNTGCA"


def test_decode_empty():
    assert decode_sequence(np.array([], dtype=np.uint8)) == ""


def test_decode_invalid_value_raises():
    with pytest.raises(ValueError, match="Cannot decode numeric value 5"):
        decode_sequence(np.array([0, 1, 5], dtype=np.uint8))


def test_roundtrip_encode_decode():
    original = "ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCC"
    encoded = encode_sequence(original)
    decoded = decode_sequence(encoded)
    assert decoded == original


def test_encode_target():
    target = "ATGCCCCAACTAAATACTAC"
    encoded = encode_target(target)
    assert len(encoded) == 20
    assert decode_sequence(encoded) == target


def test_validate_dna_sequence():
    assert validate_dna_sequence("ACGTN") is True
    assert validate_dna_sequence("acgtn") is True
    assert validate_dna_sequence("ATGCN") is True
    assert validate_dna_sequence("ATGCX") is False
    assert validate_dna_sequence("") is True
