#!/usr/bin/env bash
# Build the static GitHub Pages deploy (complete in-browser Rust via WebAssembly).
# Output goes to backend/public/ (served by the HTTP server) and pages/
# (published to GitHub Pages).
set -euo pipefail
cd "$(dirname "$0")"

echo "▸ compiling TypeScript client…"
( cd frontend && npx tsc -p tsconfig.json )

echo "▸ building Rust engine (native)…"
( cd backend && cargo build --release )

echo "▸ building Rust engine → WebAssembly…"
( cd backend && cargo build --release --target wasm32-unknown-unknown )

echo "▸ updating the served/public files…"
cp frontend/index.html backend/public/index.html
cp backend/target/wasm32-unknown-unknown/release/p2h_engine.wasm backend/public/p2h.wasm

echo "▸ updating the GitHub Pages folder (pages/)…"
cp frontend/index.html pages/index.html
cp backend/public/app.js pages/app.js
cp backend/target/wasm32-unknown-unknown/release/p2h_engine.wasm pages/p2h.wasm

echo "▸ done. Serve backend/public via the Rust HTTP server,"
echo "  or publish pages/ as a static site to GitHub Pages."
ls -la pages/
