"""Owned binary codes and explicit decoders for finite-budget channel experiments."""

import numpy as np
from scipy.special import logsumexp


def bits(values):
    value = np.asarray(values)
    if not np.all((value == 0) | (value == 1)):
        raise ValueError("Binary values required")
    return value.astype(np.uint8)


def rref(matrix):
    result = bits(matrix).copy()
    if result.ndim != 2:
        raise ValueError("Binary matrix required")
    pivots = []
    row = 0
    for col in range(result.shape[1]):
        candidates = np.flatnonzero(result[row:, col])
        if not len(candidates):
            continue
        pivot = row + candidates[0]
        result[[row, pivot]] = result[[pivot, row]]
        other = np.flatnonzero(result[:, col])
        other = other[other != row]
        result[other] ^= result[row]
        pivots.append(col)
        row += 1
        if row == len(result):
            break
    return result, pivots


def generator(check):
    reduced, pivots = rref(check)
    free = np.array([j for j in range(reduced.shape[1]) if j not in pivots], dtype=int)
    g = np.zeros((len(free), reduced.shape[1]), dtype=np.uint8)
    g[:, free] = np.eye(len(free), dtype=np.uint8)
    g[:, pivots] = reduced[: len(pivots), free].T
    if np.any((bits(check) @ g.T) % 2):
        raise RuntimeError("Nullspace construction failed")
    return g, free


def hamming_matrices():
    check = ((np.arange(1, 8)[None, :] >> np.arange(3)[:, None]) & 1).astype(np.uint8)
    g, free = generator(check)
    return check, g, free


def hamming_encode(message):
    message = bits(message)
    if message.ndim != 2 or message.shape[1] % 4:
        raise ValueError("Messages need a positive multiple of four bits")
    _, g, _ = hamming_matrices()
    return ((message.reshape(len(message), -1, 4) @ g) % 2).reshape(len(message), -1)


def hamming_decode(llr, soft=False):
    llr = np.asarray(llr, dtype=float)
    if llr.ndim != 2 or llr.shape[1] % 7:
        raise ValueError("Received codewords need a multiple of seven soft values")
    check, g, free = hamming_matrices()
    groups = llr.reshape(-1, 7)
    if soft:
        messages = ((np.arange(16)[:, None] >> np.arange(4)[None, :]) & 1).astype(np.uint8)
        words = (messages @ g) % 2
        likelihood = groups @ (1 - 2 * words.astype(float)).T
        decoded = messages[np.argmax(likelihood, axis=1)]
    else:
        words = (groups < 0).astype(np.uint8)
        syndrome = ((words @ check.T) % 2) @ (2 ** np.arange(3))
        indices = np.flatnonzero(syndrome)
        words[indices, syndrome[indices] - 1] ^= 1
        decoded = words[:, free]
    return decoded.reshape(len(llr), -1)


def convolutional_encode(message):
    """Constraint length 7, generators 133/171 octal, six zero termination bits."""
    message = bits(message)
    if message.ndim != 2 or not message.shape[1]:
        raise ValueError("Nonempty message matrix required")
    output = np.empty((len(message), 2 * (message.shape[1] + 6)), dtype=np.uint8)
    state = np.zeros(len(message), dtype=np.uint8)
    for t in range(message.shape[1] + 6):
        symbol = message[:, t] if t < message.shape[1] else 0
        register = (state.astype(np.uint16) << 1) | symbol
        for j, polynomial in enumerate([0o133, 0o171]):
            parity = np.zeros(len(message), dtype=np.uint8)
            for shift in range(7):
                if (polynomial >> shift) & 1:
                    parity ^= ((register >> shift) & 1).astype(np.uint8)
            output[:, 2 * t + j] = parity
        state = (register & 63).astype(np.uint8)
    return output


def viterbi_decode(llr):
    llr = np.asarray(llr, dtype=float)
    if llr.ndim != 2 or llr.shape[1] < 14 or llr.shape[1] % 2:
        raise ValueError("Terminated soft convolutional codewords required")
    batch, length = len(llr), llr.shape[1] // 2
    destinations = np.arange(64)
    symbol = destinations & 1
    previous0, previous1 = destinations >> 1, (destinations >> 1) + 32
    outputs = []
    for previous in [previous0, previous1]:
        register = (previous << 1) | symbol
        out = np.zeros((64, 2), dtype=float)
        for j, polynomial in enumerate([0o133, 0o171]):
            for shift in range(7):
                if (polynomial >> shift) & 1:
                    out[:, j] = np.logical_xor(out[:, j], (register >> shift) & 1)
        outputs.append(1 - 2 * out)
    metric = np.full((batch, 64), -np.inf)
    metric[:, 0] = 0
    trace = np.empty((length, batch, 64), dtype=np.uint8)
    for t in range(length):
        received = llr[:, 2 * t : 2 * t + 2]
        left = metric[:, previous0] + received @ outputs[0].T
        right = metric[:, previous1] + received @ outputs[1].T
        choose_right = right > left
        metric = np.maximum(left, right)
        trace[t] = np.where(choose_right, previous1[None, :], previous0[None, :])
    # Terminated encoder ends at known state zero.
    state = np.zeros(batch, dtype=int)
    decoded = np.empty((batch, length), dtype=np.uint8)
    for t in range(length - 1, -1, -1):
        decoded[:, t] = state & 1
        state = trace[t, np.arange(batch), state]
    return decoded[:, :-6]


def ldpc_matrices(message_bits=64, seed=991):
    """Balanced degree-three repeat/dual-diagonal accumulate construction.

    H=[A|T] with invertible lower dual-diagonal T gives an exact rate 1/2.
    Greedy check selection avoids existing check-pair co-occurrences where
    possible. This short owned construction is not an industry standard profile.
    """
    k = int(message_bits)
    if k < 8:
        raise ValueError("At least eight message bits required")
    rng = np.random.default_rng(seed)
    a = np.zeros((k, k), dtype=np.uint8)
    t = np.eye(k, dtype=np.uint8)
    t[np.arange(1, k), np.arange(k - 1)] = 1
    coexist = (t.astype(int) @ t.T) > 0
    load = t.sum(1).astype(int)
    for col in range(k):
        selected = []
        for _ in range(3):
            permitted = np.ones(k, dtype=bool)
            permitted[selected] = False
            for chosen in selected:
                permitted &= ~coexist[chosen]
            candidates = np.flatnonzero(permitted)
            if not len(candidates):
                candidates = np.array([j for j in range(k) if j not in selected])
            candidates = candidates[load[candidates] == load[candidates].min()]
            chosen = int(rng.choice(candidates))
            selected.append(chosen)
            load[chosen] += 1
        a[selected, col] = 1
        coexist[np.ix_(selected, selected)] = True
    check = np.concatenate([a, t], axis=1)
    parity = np.mod(np.cumsum(a, axis=0), 2)
    g = np.concatenate([np.eye(k, dtype=np.uint8), parity.T], axis=1)
    assert not np.any((check @ g.T) % 2)
    return check, g


def ldpc_decode(llr, check, method="sum_product", max_iter=50, normalization=0.8):
    llr = np.asarray(llr, dtype=float)
    h = bits(check)
    if llr.ndim != 2 or h.ndim != 2 or h.shape[1] != llr.shape[1]:
        raise ValueError("Soft values and check matrix must align")
    if method not in ["sum_product", "normalized_min_sum"] or max_iter < 1:
        raise ValueError("Invalid decoder method or iteration budget")
    rows, cols = np.nonzero(h)
    groups = [np.flatnonzero(rows == i) for i in range(len(h))]
    if any(len(g) < 2 for g in groups):
        raise ValueError("Check nodes need at least two edges")
    variable = np.clip(llr[:, cols], -30, 30)
    finished = np.zeros(len(llr), dtype=bool)
    iterations = np.full(len(llr), max_iter, dtype=int)
    saved = np.zeros(llr.shape, dtype=np.uint8)
    scatter_rows = np.arange(len(llr))[:, None]
    for iteration in range(1, max_iter + 1):
        messages = np.empty(variable.shape)
        for group in groups:
            v = variable[:, group]
            if method == "sum_product":
                tanh = np.tanh(v / 2)
                prefix = np.cumprod(np.c_[np.ones(len(v)), tanh], axis=1)
                suffix = np.cumprod(np.c_[tanh, np.ones(len(v))][:, ::-1], axis=1)[:, ::-1]
                product = prefix[:, :-1] * suffix[:, 1:]
                messages[:, group] = 2 * np.arctanh(np.clip(product, -1 + 1e-12, 1 - 1e-12))
            else:
                sign = np.where(v >= 0, 1, -1)
                absolute = np.abs(v)
                minimum = np.argmin(absolute, axis=1)
                first_two = np.partition(absolute, 1, axis=1)[:, :2]
                excluding = np.broadcast_to(first_two[:, :1], v.shape).copy()
                excluding[np.arange(len(v)), minimum] = first_two[:, 1]
                messages[:, group] = (
                    normalization * excluding * np.prod(sign, axis=1)[:, None] * sign
                )
        posterior = llr.copy()
        np.add.at(posterior, (scatter_rows, cols[None, :]), messages)
        hard = (posterior < 0).astype(np.uint8)
        valid = ~np.any((hard @ h.T) % 2, axis=1)
        newly_finished = valid & ~finished
        saved[newly_finished] = hard[newly_finished]
        iterations[newly_finished] = iteration
        finished |= valid
        if finished.all():
            break
        variable = np.clip(posterior[:, cols] - messages, -30, 30)
    saved[~finished] = hard[~finished]
    return saved, {"syndrome_satisfied": finished, "iterations": iterations}


def polar_transform(values):
    output = bits(values).copy()
    if output.ndim != 2 or output.shape[1] < 2 or output.shape[1] & (output.shape[1] - 1):
        raise ValueError("Polar blocklength must be a power of two")
    stride = 1
    while stride < output.shape[1]:
        for start in range(0, output.shape[1], 2 * stride):
            output[:, start : start + stride] ^= output[:, start + stride : start + 2 * stride]
        stride *= 2
    return output


def polar_information_set(blocklength, message_bits, design_erasure=0.5):
    if blocklength < 2 or blocklength & (blocklength - 1) or not 1 <= message_bits <= blocklength:
        raise ValueError("Invalid Polar blocklength or message dimension")
    if not 0 < design_erasure < 1:
        raise ValueError("Interior construction erasure probability required")
    z = np.array([design_erasure])
    while len(z) < blocklength:
        z = np.stack([2 * z - z * z, z * z], axis=1).ravel()
    return np.sort(np.argsort(z, kind="stable")[:message_bits]), z


def polar_encode(message, blocklength=128, design_erasure=0.5):
    message = bits(message)
    if message.ndim != 2:
        raise ValueError("Message matrix required")
    information, _ = polar_information_set(blocklength, message.shape[1], design_erasure)
    u = np.zeros((len(message), blocklength), dtype=np.uint8)
    u[:, information] = message
    return polar_transform(u)


def polar_decode(llr, message_bits=64, design_erasure=0.5):
    llr = np.asarray(llr, dtype=float)
    if llr.ndim != 2 or not np.isfinite(llr).all():
        raise ValueError("Finite received soft-value matrix required")
    information, _ = polar_information_set(llr.shape[1], message_bits, design_erasure)
    frozen = np.ones(llr.shape[1], dtype=bool)
    frozen[information] = False

    def recurse(values, start):
        n = values.shape[1]
        if n == 1:
            bit = (
                np.zeros((len(values), 1), dtype=np.uint8)
                if frozen[start]
                else (values < 0).astype(np.uint8)
            )
            return bit, bit
        a, b = values[:, : n // 2], values[:, n // 2 :]
        # Stable exact f(a,b), avoiding tanh saturation at large received LLRs.
        left_llr = np.logaddexp(0, a + b) - np.logaddexp(a, b)
        left_u, left_partial = recurse(left_llr, start)
        right_llr = b + (1 - 2 * left_partial.astype(float)) * a
        right_u, right_partial = recurse(right_llr, start + n // 2)
        return np.c_[left_u, right_u], np.c_[left_partial ^ right_partial, right_partial]

    u, _ = recurse(llr, 0)
    return u[:, information]


def bpsk_information(noise_variance, nodes=80):
    """Binary-input AWGN information via Gaussian quadrature, bits/real use."""
    if not np.isfinite(noise_variance) or noise_variance <= 0 or nodes < 4:
        raise ValueError("Positive noise variance and quadrature resolution required")
    x, weight = np.polynomial.hermite.hermgauss(nodes)
    received = 1 + np.sqrt(2 * noise_variance) * x
    llr = 2 * received / noise_variance
    entropy = weight @ np.logaddexp(0, -llr) / np.sqrt(np.pi) / np.log(2)
    return float(1 - entropy)


def soft_linear_decode(llr, generator_matrix):
    """Exact ML enumerator for small independent decoder tests, not large codes."""
    g = bits(generator_matrix)
    if len(g) > 16:
        raise ValueError("Exact enumeration limited to sixteen information bits")
    messages = ((np.arange(2 ** len(g))[:, None] >> np.arange(len(g))[None, :]) & 1).astype(
        np.uint8
    )
    codewords = (messages @ g) % 2
    score = np.asarray(llr) @ (1 - 2 * codewords.astype(float)).T
    chosen = np.argmax(score, axis=1)
    return messages[chosen], logsumexp(score / 2, axis=1)
