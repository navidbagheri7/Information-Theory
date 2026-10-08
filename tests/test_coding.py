"""Decoder oracles using exhaustive likelihoods and independent GF(2) algebra."""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from information_lab.coding import (
    convolutional_encode,
    hamming_decode,
    hamming_encode,
    hamming_matrices,
    ldpc_decode,
    ldpc_matrices,
    polar_decode,
    polar_encode,
    polar_information_set,
    viterbi_decode,
)
from information_lab.lossless import compress, decompress, huffman, prefix_free, shannon


@pytest.mark.parametrize(
    "method", ["huffman", "shannon", "arithmetic0", "arithmetic1", "lz77", "deflate", "zstd"]
)
def test_lossless_real_frames_and_edge_cases(method):
    rng = np.random.default_rng(9)
    examples = [
        b"",
        b"x",
        b"0" * 40000,
        bytes(range(256)),
        rng.integers(0, 256, 4096, dtype=np.uint8).tobytes(),
        b"ABAABABAAB" * 800,
    ]
    for data in examples:
        frame = compress(data, method)
        assert decompress(frame, method) == data
    if method not in ["deflate", "zstd"]:
        frame = bytearray(compress(b"test checksum", method))
        frame[16] ^= 1
        with pytest.raises(ValueError):
            decompress(frame, method)


def test_prefix_lengths_match_exhaustive_course_optimum():
    p = np.array([1 / 3, 1 / 4, 1 / 6, 1 / 12, 1 / 16, 1 / 16, 1 / 24])
    code = huffman(p)
    assert prefix_free(code) and prefix_free(shannon(p))
    cost = sum(p[i] * len(c) for i, c in code.items())
    # Exhaustive feasible sorted length assignments; optimal lengths are monotone
    # in decreasing probabilities by a swap argument, so this is an exact search.
    best = np.inf
    ordered = sorted(p, reverse=True)
    for lengths in itertools.combinations_with_replacement(range(1, 7), 7):
        if sum(2.0 ** (-ell) for ell in lengths) <= 1:
            best = min(best, np.dot(ordered, lengths))
    np.testing.assert_allclose(cost, best, atol=1e-14)
    assert huffman([0, 1]) == {1: ""}


def test_hamming_corrects_every_single_error():
    message = np.array(list(itertools.product([0, 1], repeat=4)), dtype=np.uint8)
    word = hamming_encode(message)
    for position in range(7):
        corrupted = word.copy()
        corrupted[:, position] ^= 1
        restored = hamming_decode(10 * (1 - 2 * corrupted.astype(float)))
        np.testing.assert_array_equal(restored, message)
    check, g, _ = hamming_matrices()
    assert not np.any((check @ g.T) % 2)


def test_viterbi_noisy_decisions_equal_brute_force_ml():
    k = 5
    messages = np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.uint8)
    words = []
    for message in messages:
        state, out = 0, []
        for bit in list(message) + [0] * 6:
            register = (state << 1) | int(bit)
            out.extend([(register & 0o133).bit_count() % 2, (register & 0o171).bit_count() % 2])
            state = register & 63
        words.append(out)
    words = np.array(words)
    llr = np.random.default_rng(82).normal(size=(30, 2 * (k + 6)))
    chosen = np.argmax(llr @ (1 - 2 * words).T, axis=1)
    np.testing.assert_array_equal(viterbi_decode(llr), messages[chosen])
    np.testing.assert_array_equal(convolutional_encode(messages), words)


def test_polar_noisy_sc_equals_exhaustive_synthetic_channel_decisions():
    n, k = 8, 4
    information, _ = polar_information_set(n, k)
    u = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.uint8)
    f = np.array([[1, 0], [1, 1]])
    transform = np.kron(np.kron(f, f), f)
    words = (u @ transform) % 2
    received = np.random.default_rng(72).normal(size=(21, n))
    expected = []
    for llr in received:
        score = (1 - 2 * words.astype(float)) @ llr / 2
        prefix = []
        for position in range(n):
            mask = np.all(u[:, :position] == np.array(prefix)[None, :], axis=1)
            if position not in information:
                prefix.append(0)
            else:
                # Future synthetic-channel bits remain uniform, including bits
                # that SC will freeze later; this is not a full MAP code decoder.
                likelihood0 = logsumexp(score[mask & (u[:, position] == 0)])
                likelihood1 = logsumexp(score[mask & (u[:, position] == 1)])
                prefix.append(int(likelihood1 > likelihood0))
        expected.append(np.array(prefix)[information])
    np.testing.assert_array_equal(polar_decode(received, k), expected)
    unfrozen = np.zeros(u.shape, dtype=np.uint8)
    unfrozen[:, information] = u[:, information]
    np.testing.assert_array_equal(polar_encode(u[:, information], n), (unfrozen @ transform) % 2)


def test_ldpc_noiseless_and_independent_first_bp_iteration():
    h, g = ldpc_matrices(8)
    messages = np.array(list(itertools.product([0, 1], repeat=8)), dtype=np.uint8)
    words = (messages @ g) % 2
    for method in ["sum_product", "normalized_min_sum"]:
        result, diagnostic = ldpc_decode(30 * (1 - 2 * words.astype(float)), h, method=method)
        np.testing.assert_array_equal(result[:, :8], messages)
        assert diagnostic["syndrome_satisfied"].all()
    llr = np.random.default_rng(3).normal(size=(7, 16))
    posterior = llr.copy()
    for row in h:
        neighbors = np.flatnonzero(row)
        for variable in neighbors:
            other = neighbors[neighbors != variable]
            product = np.prod(np.tanh(llr[:, other] / 2), axis=1)
            posterior[:, variable] += 2 * np.arctanh(product)
    expected = (posterior < 0).astype(np.uint8)
    actual, _ = ldpc_decode(llr, h, max_iter=1)
    np.testing.assert_array_equal(actual, expected)
