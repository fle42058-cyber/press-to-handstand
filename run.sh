#!/usr/bin/env bash
# Build the TypeScript client and run the Rust server (serves UI + API on :8080).
set -euo pipefail
cd "$(dirname "$0")"

echo "▸ compiling TypeScript client…"
( cd frontend && npx tsc -p tsconfig.json )

echo "▸ building Rust engine…"
( cd backend && cargo build --release )

echo "▸ starting server at http://0.0.0.0:8080"
cd backend
./target/release/p2h-engine
