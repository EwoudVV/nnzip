import hashlib
import os
import sys


def compress(filename):
    with open(filename, "rb") as f:
        data = f.read()
    hash_hex = hashlib.sha256(data).hexdigest()
    length = len(data)
    return hash_hex, length


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 compress.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    hash_hex, length = compress(input_file)

    with open(output_file, "w") as f:
        f.write(f"{hash_hex}\n{length}\n")

    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    print(f"original:     {original_size} bytes")
    print(f"'compressed': {compressed_size} bytes")
    print(f"hash:         {hash_hex}")
    print(f"length:       {length}")
