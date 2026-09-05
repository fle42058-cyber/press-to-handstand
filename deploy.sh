#!/usr/bin/env bash
# Publish the static GitHub Pages site to a GitHub repository.
#
# Usage:  ./deploy.sh  <owner>/<repo>          (e.g. ./deploy.sh you/p2h-handstand)
#         ./deploy.sh  you/p2h-handstand  gh-pages
#
# It builds backend/public (Rust+wasm+ts) and publishes the CONTENTS of
# backend/public/ to the branch you choose (default gh-pages). Because the
# engine runs entirely in WebAssembly, the site is fully static — no server.
set -euo pipefail
cd "$(dirname "$0")"

REPO="${1:?usage: ./deploy.sh <owner>/<repo> [branch]}"
BRANCH="${2:-gh-pages}"
REMOTE="https://github.com/${REPO}.git"
TMP="$(mktemp -d)"

echo "▸ building site (ts + Rust + wasm)…"
./build.sh

echo "▸ publishing backend/public/ → ${REPO}#${BRANCH}"

# Stage the deployable files in a git tree on the target branch.
cd "$TMP"
git init -q
git checkout -q --orphan "$BRANCH"
git config user.name "p2h deploy"
git config user.email "deploy@example.com"
cp -R /home/user/p2h/backend/public/. .
touch .nojekyll
git add -A
git commit -q -m "publish press-to-handstand (Rust->WASM, live in-browser evolution)"
echo "▸ pushing to ${REMOTE} (${BRANCH})…"
git push -q --force "$REMOTE" HEAD:"$BRANCH"

echo "▸ done. Enable Pages on this repo pointing at the ${BRANCH} branch."
echo "  https://github.com/${REPO}/settings/pages"
rm -rf "$TMP"
