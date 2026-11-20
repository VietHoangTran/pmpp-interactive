#!/bin/sh
# Build and test every lab on every backend this machine has, and write a
# report to reports/verify-<time>-<os>-<backends>.md (plus a .json twin).
# Pass --quick to skip the benchmarks.
set -e
cd "$(dirname "$0")"
exec python3 labs/pmpp.py verify "$@"
