import sys


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 decompress_index.py <compressed_file> <output_file>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        data = f.read()

    length = data[0]
    index = int.from_bytes(data[1:], "big") if data[1:] else 0
    recovered = index.to_bytes(length, "big")

    with open(sys.argv[2], "wb") as f:
        f.write(recovered)

    print(f"recovered: {recovered!r}")
