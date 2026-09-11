#!/usr/bin/env python3
"""Stream benchmark diagnostics without mixing progress into result CSVs."""

import argparse
import codecs
from contextlib import ExitStack
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


def run(command, *, name, output, errors=None, interval=30.0):
    started = time.monotonic()

    def report(message):
        print(f"[{name}] {message} (elapsed {time.monotonic() - started:.1f}s)",
              flush=True)

    report("START")
    with ExitStack() as stack:
        destination = stack.enter_context(Path(output).open("wb"))
        # With a separate stderr file, stdout is CSV and is never echoed.
        # Without one, combine both streams into a live validation log.
        diagnostics = (stack.enter_context(Path(errors).open("wb"))
                       if errors is not None else destination)
        process = stack.enter_context(subprocess.Popen(
            command, stdout=destination if errors is not None else subprocess.PIPE,
            stderr=subprocess.PIPE if errors is not None else subprocess.STDOUT))
        stream = process.stderr if errors is not None else process.stdout
        selector = stack.enter_context(selectors.DefaultSelector())
        selector.register(stream, selectors.EVENT_READ)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        next_heartbeat = started + interval
        try:
            while selector.get_map() or process.poll() is None:
                timeout = max(0.0, next_heartbeat - time.monotonic())
                if selector.get_map():
                    events = selector.select(timeout)
                else:
                    # A child may close its output before it exits. Waiting
                    # on it avoids either spinning or delaying completion.
                    try:
                        process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        pass
                    events = []
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    diagnostics.write(chunk)
                    diagnostics.flush()
                    print(decoder.decode(chunk), end="", flush=True)
                if time.monotonic() >= next_heartbeat:
                    report("RUNNING; waiting for the current command")
                    next_heartbeat = time.monotonic() + interval
            print(decoder.decode(b"", final=True), end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code < 0:
        report(f"FAILED: terminated by {signal.Signals(-code).name} ({-code})")
        return 128 - code
    report("COMPLETE" if code == 0 else f"FAILED: exit code {code}")
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--interval", type=float,
                        default=os.environ.get("CLQR_PROGRESS_INTERVAL_SECONDS", "30"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    return run(command, name=args.name, output=args.stdout,
               errors=args.stderr, interval=args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
