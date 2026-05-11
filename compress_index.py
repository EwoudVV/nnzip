import os
import sys


def compress(filename):
    with open(filename, "rb") as f:
        data = f.read()
    length = len(data)
    # the index of `data` in itertools.product(range(256), repeat=length)
    # is just `data` interpreted as a big-endian integer
    index = int.from_bytes(data, "big") if data else 0
    return length, index


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 compress_index.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    length, index = compress(input_file)

    # store length as 1 byte, then the index with leading zero bytes stripped
    index_bytes = index.to_bytes((index.bit_length() + 7) // 8, "big") if index else b""

    with open(output_file, "wb") as f:
        f.write(length.to_bytes(1, "big"))
        f.write(index_bytes)

    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    print(f"original:     {original_size} bytes")
    print(f"'compressed': {compressed_size} bytes")
    print(f"length:       {length}")
    print(f"index:        {index:,}")
