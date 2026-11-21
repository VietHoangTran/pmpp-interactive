#!/bin/sh
# Build and test every lab on every backend this machine has, and write a
# report to reports/verify-<time>-<os>-<backends>.md (plus a .json twin).
# Pass --quick to skip the benchmarks, --reps N to run each benchmark N times (every
# repetition is recorded), --full to keep sanitizer reports in full. PMPP_CMAKE_ARGS,
# PMPP_BUILD_DIR and PMPP_ONLY apply as for labs/pmpp.py (see labs/README.md).
set -e
cd "$(dirname "$0")"
# pmpp.py's own tests first (starter history): a broken runner would make the report meaningless.
python3 -m unittest discover -s labs/tests -q
exec python3 labs/pmpp.py verify "$@"
