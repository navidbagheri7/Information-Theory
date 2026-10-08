"""Real lossless bitstreams: prefix codes, adaptive arithmetic, LZ77 and standard codecs.

Owned formats carry decoded length and CRC32. Prefix formats transmit all 256
symbol frequencies; arithmetic models learn online from shared unit pseudocounts.
The explicitly specified LZ77 format is not a guessed transcription of the
course's class-only integer-code convention.
"""

import heapq
import math
import struct
import zlib
from collections import defaultdict, deque

import numpy as np

HEADER = struct.Struct(">4sBQI")
METHODS = {"huffman": 1, "shannon": 2, "arithmetic0": 3, "arithmetic1": 4, "lz77": 5}


class BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.value = 0
        self.filled = 0
        self.bits = 0

    def bit(self, value):
        self.value = (self.value << 1) | int(value)
        self.filled += 1
        self.bits += 1
        if self.filled == 8:
            self.buffer.append(self.value)
            self.value = self.filled = 0

    def integer(self, value, width):
        for shift in range(width - 1, -1, -1):
            self.bit((value >> shift) & 1)

    def gamma(self, value):
        if value < 1:
            raise ValueError("Gamma integers are positive")
        width = value.bit_length()
        self.integer(0, width - 1)
        self.integer(value, width)

    def finish(self):
        if self.filled:
            self.buffer.append(self.value << (8 - self.filled))
            self.value = self.filled = 0
        return bytes(self.buffer)


class BitReader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.position = 0

    def bit(self):
        if self.position >= len(self.data) * 8:
            raise ValueError("Truncated bitstream")
        value = (self.data[self.position // 8] >> (7 - self.position % 8)) & 1
        self.position += 1
        return value

    def integer(self, width):
        value = 0
        for _ in range(width):
            value = (value << 1) | self.bit()
        return value

    def gamma(self):
        zeros = 0
        while self.bit() == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("Invalid gamma length")
        return (1 << zeros) | self.integer(zeros)


def huffman(weights):
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or not len(weights)
        or np.any(weights < 0)
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Finite nonnegative symbol weights required")
    heap = [(float(v), i, i) for i, v in enumerate(weights) if v > 0]
    if not heap:
        raise ValueError("At least one symbol must have positive weight")
    heapq.heapify(heap)
    serial = len(weights)
    while len(heap) > 1:
        a, _, left = heapq.heappop(heap)
        b, _, right = heapq.heappop(heap)
        heapq.heappush(heap, (a + b, serial, (left, right)))
        serial += 1
    codes = {}

    def visit(tree, prefix):
        if isinstance(tree, int):
            # A singleton alphabet needs no payload bits: the frame supplies n.
            codes[tree] = prefix
        else:
            visit(tree[0], prefix + "0")
            visit(tree[1], prefix + "1")

    visit(heap[0][2], "")
    return codes


def shannon(weights):
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or not len(weights)
        or np.any(weights < 0)
        or not np.isfinite(weights).all()
        or weights.sum() <= 0
    ):
        raise ValueError("Valid positive-total symbol weights required")
    p = weights / weights.sum()
    lengths = sorted((math.ceil(-math.log2(v)), i) for i, v in enumerate(p) if v > 0)
    code, previous = 0, lengths[0][0]
    result = {}
    for length, symbol in lengths:
        code <<= length - previous
        result[symbol] = format(code, f"0{length}b") if length else ""
        code += 1
        previous = length
    return result


def prefix_free(codes):
    values = sorted(codes.values())
    return all(not b.startswith(a) for a, b in zip(values, values[1:], strict=False))


def _frame(method, raw, payload):
    return HEADER.pack(b"ITL1", METHODS[method], len(raw), zlib.crc32(raw)) + payload


def _unframe(method, encoded, limit_bytes):
    if len(encoded) < HEADER.size:
        raise ValueError("Truncated frame")
    magic, identifier, size, crc = HEADER.unpack_from(encoded)
    if magic != b"ITL1" or identifier != METHODS[method] or size > limit_bytes:
        raise ValueError("Invalid format, method or decoded-size limit")
    return size, crc, encoded[HEADER.size :]


def _prefix_encode(data, method):
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    if not len(data):
        counts[0] = 1
    codes = huffman(counts) if method == "huffman" else shannon(counts)
    bits = BitWriter()
    for symbol in data:
        code = codes[symbol]
        if code:
            bits.integer(int(code, 2), len(code))
    table = struct.pack(">256I", *counts)
    return _frame(method, data, table + bits.finish())


def _prefix_decode(payload, size, method):
    if len(payload) < 1024:
        raise ValueError("Missing prefix-code model")
    counts = struct.unpack_from(">256I", payload)
    codes = huffman(counts) if method == "huffman" else shannon(counts)
    trie = {}
    for symbol, code in codes.items():
        node = trie
        for bit in code:
            node = node.setdefault(int(bit), {})
        node["symbol"] = symbol
    reader = BitReader(payload[1024:])
    out = bytearray()
    for _ in range(size):
        node = trie
        while "symbol" not in node:
            bit = reader.bit()
            if bit not in node:
                raise ValueError("Invalid prefix-code path")
            node = node[bit]
        out.append(node["symbol"])
    return bytes(out)


def _arithmetic(data, order, decode_size=None):
    """32-bit integer arithmetic coder with E1/E2/E3 renormalization."""
    counts = np.ones((257 if order else 1, 256), dtype=np.int64)
    context = 256 if order else 0
    low, high = 0, 2**32 - 1
    half, quarter, three_quarters = 2**31, 2**30, 3 * 2**30
    decode = decode_size is not None
    pending = 0
    writer = BitWriter()
    reader = BitReader(data) if decode else None
    value = reader.integer(32) if decode else None
    output = bytearray()

    def emit(bit):
        nonlocal pending
        writer.bit(bit)
        for _ in range(pending):
            writer.bit(1 - bit)
        pending = 0

    for index in range(decode_size if decode else len(data)):
        row = counts[context]
        cumulative = np.r_[0, np.cumsum(row)]
        total = int(cumulative[-1])
        interval = high - low + 1
        if decode:
            slot = ((value - low + 1) * total - 1) // interval
            symbol = int(np.searchsorted(cumulative, slot, side="right") - 1)
            if not 0 <= symbol < 256:
                raise ValueError("Invalid arithmetic state")
            output.append(symbol)
        else:
            symbol = data[index]
        high = low + interval * int(cumulative[symbol + 1]) // total - 1
        low += interval * int(cumulative[symbol]) // total
        while True:
            if high < half:
                if not decode:
                    emit(0)
            elif low >= half:
                if decode:
                    value -= half
                else:
                    emit(1)
                low -= half
                high -= half
            elif low >= quarter and high < three_quarters:
                if decode:
                    value -= quarter
                else:
                    pending += 1
                low -= quarter
                high -= quarter
            else:
                break
            low *= 2
            high = 2 * high + 1
            if decode:
                value = 2 * value + reader.bit()
        row[symbol] += 1
        if total + 1 >= 32768:
            row[:] = (row + 1) // 2
        context = symbol if order else 0
    if decode:
        return bytes(output)
    pending += 1
    emit(0 if low < quarter else 1)
    # Physical 32-bit lookahead is counted in the complete file size.
    return writer.finish() + bytes(4)


def _lz_encode(data, window):
    if not 2 <= window <= 65535:
        raise ValueError("LZ window must lie in [2,65535]")
    writer = BitWriter()
    width = (window - 1).bit_length()
    index = defaultdict(deque)
    recent = deque()
    position = 0
    while position < len(data):
        while recent and recent[0][1] < position - window:
            old_key, old_position = recent.popleft()
            old_list = index[old_key]
            if old_list and old_list[0] == old_position:
                old_list.popleft()
            if not old_list:
                del index[old_key]
        best_length, best_distance = 0, 0
        key = data[position : position + 3]
        candidates = index[key]
        while candidates and candidates[0] < position - window:
            candidates.popleft()
        maximum = min(258, len(data) - position)
        for previous in list(candidates)[-256:][::-1]:
            length = 0
            while length < maximum and data[previous + length] == data[position + length]:
                length += 1
            if length > best_length:
                best_length, best_distance = length, position - previous
            if length == maximum:
                break
        if best_length >= 3:
            writer.bit(1)
            writer.integer(best_distance - 1, width)
            writer.gamma(best_length)
            consumed = best_length
        else:
            writer.bit(0)
            writer.integer(data[position], 8)
            consumed = 1
        for previous in range(position, position + consumed):
            if previous + 3 <= len(data):
                old_key = data[previous : previous + 3]
                index[old_key].append(previous)
                recent.append((old_key, previous))
        position += consumed
    return struct.pack(">H", window) + writer.finish()


def _lz_decode(payload, size):
    if len(payload) < 2:
        raise ValueError("Missing LZ window")
    window = struct.unpack_from(">H", payload)[0]
    if window < 2:
        raise ValueError("Invalid LZ window")
    reader = BitReader(payload[2:])
    width = (window - 1).bit_length()
    out = bytearray()
    while len(out) < size:
        if reader.bit() == 0:
            out.append(reader.integer(8))
        else:
            distance = reader.integer(width) + 1
            length = reader.gamma()
            if distance > min(window, len(out)) or length < 3 or len(out) + length > size:
                raise ValueError("Invalid LZ reference")
            for _ in range(length):
                out.append(out[-distance])
    return bytes(out)


def compress(data, method, window=4096):
    data = bytes(data)
    if method in ["huffman", "shannon"]:
        return _prefix_encode(data, method)
    if method in ["arithmetic0", "arithmetic1"]:
        return _frame(method, data, _arithmetic(data, int(method[-1])))
    if method == "lz77":
        return _frame(method, data, _lz_encode(data, window))
    if method == "deflate":
        return zlib.compress(data, level=9)
    if method == "zstd":
        import zstandard

        return zstandard.ZstdCompressor(level=9, write_checksum=True).compress(data)
    raise ValueError("Unknown codec")


def decompress(encoded, method, limit_bytes=64 * 1024**2):
    if method == "deflate":
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(encoded, limit_bytes + 1)
        if len(decoded) > limit_bytes or decoder.unconsumed_tail or not decoder.eof:
            raise ValueError("Incomplete stream or decoded-size limit exceeded")
        return decoded
    if method == "zstd":
        import zstandard

        decoded = zstandard.ZstdDecompressor().decompress(encoded, max_output_size=limit_bytes)
        if len(decoded) > limit_bytes:
            raise ValueError("Decoded-size limit exceeded")
        return decoded
    if method not in METHODS:
        raise ValueError("Unknown codec")
    size, crc, payload = _unframe(method, encoded, limit_bytes)
    if method in ["huffman", "shannon"]:
        decoded = _prefix_decode(payload, size, method)
    elif method in ["arithmetic0", "arithmetic1"]:
        decoded = _arithmetic(payload, int(method[-1]), decode_size=size)
    else:
        decoded = _lz_decode(payload, size)
    if zlib.crc32(decoded) != crc:
        raise ValueError("Decoded checksum mismatch")
    return decoded
