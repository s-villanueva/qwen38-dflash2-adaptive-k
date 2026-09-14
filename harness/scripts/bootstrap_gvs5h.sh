#!/usr/bin/env bash
# Fetch the pinned MIT-licensed dependency separately; no upstream data is copied
# into this repository.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <empty-dependency-directory>" >&2
  exit 64
fi

target=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
commit=0bc99eae8c6ef0d49a91808272c60928113da1ce
patch_file="$repo_root/harness/patches/gvs5h-local-qwen.patch"

if [[ -e "$target" && -n "$(find "$target" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing non-empty target: $target" >&2
  exit 65
fi
mkdir -p "$target"
git clone --no-checkout https://github.com/slee-persis/GVS5H.git "$target"
git -C "$target" checkout --detach "$commit"
(
  cd "$target/codebase/v2-current"
  git apply --check "$patch_file"
  git apply "$patch_file"
)
printf 'Prepared GVS5H at %s\n' "$target"
