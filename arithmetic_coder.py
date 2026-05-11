"""
Simple bit-level arithmetic coder, portable across Python and JavaScript.

The bit-level output is fully determined by:
  - 32-bit precision (PRECISION = 32)
  - CDF arrays of integer cumulative counts, with total <= 2^16

Both sides (Python encoder, JS decoder) must use identical CDFs and totals.

References: Witten, Neal, Cleary (1987), "Arithmetic Coding for Data Compression."
"""

PRECISION = 32
TOP = 1 << PRECISION
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
THREE_QUARTER = 3 * QUARTER
MASK = TOP - 1
MAX_TOTAL = 1 << 16  # keeps (range * cdf_high) within 48 bits for JS float64


class ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = MASK
        self.pending = 0
        self.bits = bytearray()  # appended as raw 0/1 ints

    def _emit_bit(self, bit):
        self.bits.append(bit)
        for _ in range(self.pending):
            self.bits.append(1 - bit)
        self.pending = 0

    def encode(self, cdf_low, cdf_high, total):
        """Encode a symbol given its CDF range [cdf_low, cdf_high) and total."""
        assert 0 <= cdf_low < cdf_high <= total <= MAX_TOTAL
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cdf_high) // total - 1
        self.low = self.low + (rng * cdf_low) // total

        while True:
            if self.high < HALF:
                self._emit_bit(0)
            elif self.low >= HALF:
                self._emit_bit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK

    def finish(self):
        """Flush state, return (bytes, bit_count)."""
        self.pending += 1
        if self.low < QUARTER:
            self._emit_bit(0)
        else:
            self._emit_bit(1)

        bit_count = len(self.bits)
        padding = (8 - bit_count % 8) % 8
        out = bytearray((bit_count + padding) // 8)
        for i, b in enumerate(self.bits):
            if b:
                out[i // 8] |= 1 << (7 - (i % 8))
        return bytes(out), bit_count


class ArithmeticDecoder:
    def __init__(self, data, bit_count):
        self.data = data
        self.bit_count = bit_count
        self.pos = 0
        self.low = 0
        self.high = MASK
        self.value = 0
        for _ in range(PRECISION):
            self.value = ((self.value << 1) | self._read_bit()) & MASK

    def _read_bit(self):
        if self.pos >= self.bit_count:
            return 0
        byte_idx = self.pos // 8
        bit_idx = 7 - (self.pos % 8)
        self.pos += 1
        return (self.data[byte_idx] >> bit_idx) & 1

    def _renormalize_after_decode(self, cdf_low, cdf_high, total):
        rng = self.high - self.low + 1
        self.high = self.low + (rng * cdf_high) // total - 1
        self.low = self.low + (rng * cdf_low) // total

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.value -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK
            self.value = ((self.value << 1) | self._read_bit()) & MASK

    def decode_cdf(self, cdf, total):
        """cdf: list of length n+1 with cdf[0]=0, cdf[n]=total.
        Returns symbol s in [0, n) s.t. cdf[s] <= value < cdf[s+1]."""
        rng = self.high - self.low + 1
        v = ((self.value - self.low + 1) * total - 1) // rng
        # linear search through the small alphabet
        s = 0
        while s + 1 < len(cdf) and cdf[s + 1] <= v:
            s += 1
        self._renormalize_after_decode(cdf[s], cdf[s + 1], total)
        return s

    def decode_uniform(self, total):
        """Decode a symbol assuming uniform distribution over [0, total)."""
        rng = self.high - self.low + 1
        s = ((self.value - self.low + 1) * total - 1) // rng
        self._renormalize_after_decode(s, s + 1, total)
        return int(s)


def cdf_from_quantized(quantized):
    """Given list of integer counts, return cumulative CDF [0, c0, c0+c1, ...]."""
    cdf = [0]
    for q in quantized:
        cdf.append(cdf[-1] + q)
    return cdf


# Quantize a float probability distribution to integer counts summing to `total`.
def quantize_probs(probs, total):
    """Returns list of integer counts (length = len(probs)) summing to `total`,
    each >= 1, approximating the float probabilities."""
    raw = [p * total for p in probs]
    q = [max(1, int(round(r))) for r in raw]
    diff = total - sum(q)
    while diff != 0:
        if diff > 0:
            # add to the largest entry
            idx = max(range(len(q)), key=lambda i: q[i])
            q[idx] += 1
            diff -= 1
        else:
            # subtract from the largest entry (where there's slack)
            idx = max(range(len(q)), key=lambda i: q[i])
            if q[idx] > 1:
                q[idx] -= 1
                diff += 1
            else:
                # everyone is 1; shouldn't happen unless total < len(probs)
                break
    return q


if __name__ == "__main__":
    # Self-test: round-trip a sequence of symbols.
    import random

    probs = [0.55, 0.15, 0.08, 0.04, 0.03, 0.15]
    quant = quantize_probs(probs, 16384)
    cdf = cdf_from_quantized(quant)
    total = cdf[-1]
    print(f"quantized: {quant}, sum={sum(quant)}")
    print(f"cdf: {cdf}, total={total}")

    random.seed(0)
    symbols = [
        random.choices(range(len(probs)), weights=probs, k=1)[0] for _ in range(100)
    ]

    enc = ArithmeticEncoder()
    for s in symbols:
        enc.encode(cdf[s], cdf[s + 1], total)
    data, bit_count = enc.finish()
    print(f"encoded {len(symbols)} symbols into {bit_count} bits ({len(data)} bytes)")
    expected_bits = -sum(probs[s] for s in symbols)  # H crude estimate
    print(f"  (theoretical lower bound ~ {sum(-(__import__('math').log2(probs[s])) for s in symbols):.1f} bits)")

    dec = ArithmeticDecoder(data, bit_count)
    decoded = [dec.decode_cdf(cdf, total) for _ in symbols]
    assert decoded == symbols, "MISMATCH"
    print("CDF round-trip OK")

    # uniform test
    enc2 = ArithmeticEncoder()
    values = [random.randrange(1000) for _ in range(50)]
    for v in values:
        enc2.encode(v, v + 1, 1000)
    d2, bc2 = enc2.finish()
    print(f"uniform 50 symbols / 1000 -> {bc2} bits ({len(d2)} bytes)")
    dec2 = ArithmeticDecoder(d2, bc2)
    rec = [dec2.decode_uniform(1000) for _ in values]
    assert rec == values, "UNIFORM MISMATCH"
    print("uniform round-trip OK")
