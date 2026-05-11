import hashlib
import itertools
import sys
import time

IS_TTY = sys.stdout.isatty()


def decompress(target_hash, length):
    total = 256 ** length
    print(f"target hash: {target_hash}")
    print(f"length:      {length} bytes")
    print(f"search space: {total:,} possible files\n")

    start = time.time()
    last_update = start
    last_index = 0
    milestone_step = max(total // 20, 1)  # every 5%
    next_milestone = milestone_step

    for index, combo in enumerate(itertools.product(range(256), repeat=length)):
        candidate = bytes(combo)
        if hashlib.sha256(candidate).hexdigest() == target_hash:
            if IS_TTY:
                sys.stdout.write("\r\033[K")
            elapsed = time.time() - start
            print(f"\n>>> MATCH FOUND <<<")
            print(f"index:   {index:,}")
            print(f"file:    {candidate!r}")
            print(f"hex:     {candidate.hex()}")
            print(f"elapsed: {elapsed:.2f}s")
            return candidate, index

        now = time.time()

        if index >= next_milestone:
            elapsed = now - start
            rate = index / elapsed if elapsed > 0 else 0
            if IS_TTY:
                sys.stdout.write("\r\033[K")
            print(f"  [{100*index/total:5.1f}%] checked {index:>15,} | {rate/1e6:5.2f}M/s | {elapsed:5.1f}s elapsed")
            while index >= next_milestone:
                next_milestone += milestone_step
            last_update = now
            last_index = index
        elif IS_TTY and now - last_update >= 0.1:
            elapsed = now - start
            rate = (index - last_index) / (now - last_update)
            pct = 100 * index / total
            remaining = (total - index) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r\033[K[{pct:5.2f}%] {index:>15,}/{total:,} | trying {candidate.hex()} | {rate/1e6:5.2f}M/s | ETA {remaining:5.0f}s"
            )
            sys.stdout.flush()
            last_update = now
            last_index = index

    return None, -1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 decompress.py <compressed_file> <output_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        target_hash = f.readline().strip()
        length = int(f.readline().strip())

    result, index = decompress(target_hash, length)

    if result is None:
        print("no match found")
        sys.exit(1)

    with open(sys.argv[2], "wb") as f:
        f.write(result)
