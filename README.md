# Press → Handstand · Self-Evolving Coach

A mobile-first fitness web app where a **Rust** engine runs a **genetic algorithm**
that evolves a small **neural-network "coach"** to optimise a training pathway
toward the **press to handstand (P2H)**. The frontend is **TypeScript** and
visualises the live evolution plus a personalised week-by-week programme.

> **No AI / ML libraries** — everything is hand-written: a tiny feed-forward
> network, a genetic algorithm (tournament selection, crossover, Gaussian
> mutation, elitism), a biomechanical-style progression simulator and a minimal
> std-only HTTP server. Pure algorithms.

---

## Why it's interesting

A full **press to handstand** is a strict skill: you need shoulder flexion,
wrist prep, hamstring mobility, hip-flexor *compression*, straight-arm push
strength, core/planche hold, balance and scapular control — **all at once**.
So the engine models the athlete as **8 capacities**, and **10 progression
drills** that each train specific capacities and gate on prerequisites.

A neural network coach reads the athlete's 8 capacities each week and outputs
a **focus distribution** over the drills. A simulator applies that focus to
grow the right capacities while *penalising overreach* (injury). The **fitness**
is how far/fast the coach takes the athlete toward the press, minus injury
cost. The **genetic algorithm** evolves the network weights to maximise that
fitness across a set of synthetic athlete profiles.

The recommendation is the better of:
- the **evolved neural coach**, and
- a deterministic **greedy "raise-the-floor"** algorithm (a reliable, no-ML
  baseline that always produces a sensible curriculum).

That guarantees a good plan while the network keeps self-optimising.

---

## What the UI shows

- **Evolution** panel: generation, best fitness, population size, and a live
  fitness curve as the GA improves the coach. Pause/resume.
- **My Pathway** panel: rate your 8 capacities (0–100), then the engine returns
  a **N-week programme** (consecutive drill blocks), a week-by-week attainment
  curve, weeks-to-press, injury counts and a start→end capacity build.

---

## Repository layout

```
p2h/
├── backend/                 # Rust engine (std-only, no extern crates)
│   ├── src/
│   │   ├── main.rs          # HTTP server + JSON API + static file serving
│   │   ├── nn.rs            # tiny feed-forward network (flat genome)
│   │   ├── ga.rs            # genetic algorithm (selection/crossover/mutation)
│   │   ├── evo.rs           # orchestrator: population, fitness, best-so-far
│   │   ├── model.rs         # capacities, skills, progression simulator, greedy
│   │   └── rng.rs           # xorshift64* PRNG (no rand crate)
│   ├── examples/bench.rs    # convergence/QA harness
│   └── public/              # built web UI (index.html + app.js)
└── frontend/                # TypeScript client
    ├── src/main.ts
    └── tsconfig.json
```

## Run

```bash
# 1. Build & start the Rust server (serves the web UI + API on :8080)
cd backend
cargo build --release
./target/release/p2h-engine
#   → http://0.0.0.0:8080

# 2. Rebuild the TypeScript client (after editing frontend/src)
cd frontend
npx tsc -p tsconfig.json        # outputs backend/public/app.js
```

Then open `http://localhost:8080` (mobile-width friendly).

## API

| Method | Path            | Description                                          |
| ------ | --------------- | ---------------------------------------------------- |
| GET    | `/api/meta`     | capacities, skills, horizon, press threshold         |
| GET    | `/api/evolution`| generation, best fitness, population, history[]      |
| POST   | `/api/recommend`| `{"caps":[...8]}` → programme, blocks, trace, source |
| POST   | `/api/control`  | `true`/`false` → pause/resume evolution              |

## The algorithm (brief)

- **Capacities** — 8 attributes in 0..1, e.g. `Shoulder flexion`, `Compression`.
- **Skills / drills** — 10, each `{train: [(cap, weight)], rate, prereq, danger}`.
- **Readiness** — 0..1 how ready an athlete is for a drill (min over prerequisite caps).
- **Gain** — `focus · rate · weight · (1−cap) · readiness` per capacity per week,
  scaled by a recovery factor that drops with overload.
- **Injury** — accumulates when a drill is done without its prerequisites; an
  event regresses the over-trained capacities.
- **Attainment (fitness)** — `0.38·geometricMean(caps) + 0.62·min(caps)`; the
  weak link dominates because a press needs *all* attributes dialled in.
- **Fitness score** — `100·attainment + 40·progress + reachPress·earlyBonus − 18·injuries`.
- **GA** — rank + tournament selection, uniform crossover, Gaussian mutation,
  elitism, occasional random restarts. Population evolved in a background thread.

## Notes

- `target/`, `node_modules/`, and build artifacts are regenerated and not
  needed to run (Rust has **zero** external crates; TypeScript only needs `tsc`
  to compile).
