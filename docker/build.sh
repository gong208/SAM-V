#!/usr/bin/env bash
# Build the SAM-V dependency image (default tag sam_v:latest).
#
# If you build behind a proxy, export HTTP_PROXY / HTTPS_PROXY / NO_PROXY first;
# they are forwarded to the build only when set.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-sam_v:latest}"

proxy_args=()
for v in HTTP_PROXY HTTPS_PROXY NO_PROXY; do
  lower="$(echo "$v" | tr '[:upper:]' '[:lower:]')"
  val="${!v:-${!lower:-}}"
  [ -n "$val" ] && proxy_args+=(--build-arg "$v=$val")
done

docker build -t "$IMAGE" -f "$HERE/Dockerfile" "${proxy_args[@]}" "$ROOT"
echo "Built $IMAGE"
