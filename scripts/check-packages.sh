#!/usr/bin/env bash
set -euo pipefail

root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
AC_FOUNDATION_BUILD_DIR="${AC_FOUNDATION_CHECK_BUILD_DIR:-$root/local/check-dist}" \
  "$root/scripts/build-packages.sh"
