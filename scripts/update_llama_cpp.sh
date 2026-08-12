#!/usr/bin/env bash
# Sync the llama.cpp fork's master with upstream and bump the submodule here.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sub="$root/llama.cpp"

git -C "$sub" fetch upstream --prune
git -C "$sub" checkout master
git -C "$sub" merge --ff-only upstream/master
git -C "$sub" push origin master

git -C "$root" add llama.cpp
if ! git -C "$root" diff --cached --quiet -- llama.cpp; then
	git -C "$root" commit -m "llama.cpp: sync fork master with upstream" -- llama.cpp
	git -C "$root" push origin main
fi

#
# For rebasing my llama.cpp PR
#
# cd llama.cpp
# git switch <branch name of my PR>
# git rebase upstream/master
# # inspect range-diff, build, test
# git push --force-with-lease origin <branch name of my PR>
