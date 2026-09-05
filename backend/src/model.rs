//! Domain model for building a *press to handstand* (P2H).
//!
//! The engine models an athlete as a vector of physical *capacities* (0..1).
//! Each progression *skill* trains specific capacities and asks for a set of
//! prerequisite capacities. The neural coach outputs a "focus" distribution
//! over the skills; the simulator below turns that focus into weekly capacity
//! gains, overload/injury risk, and an overall press-to-handstand attainment
//! score. That score is what evolution optimizes.

// ---------------------------------------------------------------------------
// Capacities (indices)
// ---------------------------------------------------------------------------
pub const SHOULDER_FLEX: usize = 0;
pub const WRIST_FLEX: usize = 1;
pub const HAMSTRING: usize = 2;
pub const COMPRESSION: usize = 3;
pub const PUSH_STRENGTH: usize = 4;
pub const CORE_HOLD: usize = 5;
pub const BALANCE: usize = 6;
pub const SCRAP_BACK: usize = 7;

pub const N_CAP: usize = 8;

pub const CAPACITY_NAMES: [&str; N_CAP] = [
    "Shoulder flexion",
    "Wrist prep",
    "Hamstring mobility",
    "Compression (hip flexor)",
    "Straight-arm push strength",
    "Core / planche hold",
    "Balance & body line",
    "Scapular (upper back)",
];

pub const CAPACITY_HINT: [&str; N_CAP] = [
    "Arms overhead, elbows locked (wall slides / cuban press)",
    "Palms flat, wrists bent under load",
    "Standing pike fold depth",
    "Bringing knees/legs to chest",
    "Locked-elbow pressing (planche push, dips)",
    "Hollow body / planche lean hold",
    "Handstand balance & line",
    "Lat / rhomboid engagement (scapular pull)",
];

// ---------------------------------------------------------------------------
// Skills (indices) — the progression ladder toward the full press.
// ---------------------------------------------------------------------------
pub const SKILL_COUNT: usize = 10;

pub const SKILL_NAMES: [&str; SKILL_COUNT] = [
    "Wall slides",
    "Wrist prep",
    "Standing pike fold",
    "Elevated pike hold",
    "Wall walk / hip tap",
    "Tuck planche lean",
    "L-sit compression",
    "Straddle press negative",
    "Pike press (feet to hands)",
    "Full press to handstand",
];

// (capacity_index, weight) — what each skill builds.
// rate      — base learning rate per unit focus in a week.
// prereq    — (capacity_index, minimum) every prerequisite must be reached.
// danger    — injury risk coefficient when the skill is pursued *without*
//             its prerequisites in place.
pub struct Skill {
    pub train: &'static [(usize, f32)],
    pub rate: f32,
    pub prereq: &'static [(usize, f32)],
    pub danger: f32,
}

pub const SKILLS: [Skill; SKILL_COUNT] = [
    Skill {
        train: &[(SHOULDER_FLEX, 0.8), (SCRAP_BACK, 0.2)],
        rate: 0.42,
        prereq: &[],
        danger: 0.02,
    },
    Skill {
        train: &[(WRIST_FLEX, 0.9), (SCRAP_BACK, 0.1)],
        rate: 0.40,
        prereq: &[],
        danger: 0.03,
    },
    Skill {
        train: &[(HAMSTRING, 0.7), (COMPRESSION, 0.3)],
        rate: 0.40,
        prereq: &[],
        danger: 0.03,
    },
    Skill {
        train: &[(COMPRESSION, 0.6), (CORE_HOLD, 0.25), (BALANCE, 0.15)],
        rate: 0.30,
        prereq: &[(HAMSTRING, 0.35), (SHOULDER_FLEX, 0.35)],
        danger: 0.05,
    },
    Skill {
        train: &[
            (PUSH_STRENGTH, 0.35),
            (BALANCE, 0.3),
            (SHOULDER_FLEX, 0.15),
            (SCRAP_BACK, 0.1),
            (WRIST_FLEX, 0.1),
        ],
        rate: 0.30,
        prereq: &[(WRIST_FLEX, 0.35), (SHOULDER_FLEX, 0.4)],
        danger: 0.07,
    },
    Skill {
        train: &[(CORE_HOLD, 0.5), (PUSH_STRENGTH, 0.4), (SCRAP_BACK, 0.1)],
        rate: 0.26,
        prereq: &[(WRIST_FLEX, 0.4), (CORE_HOLD, 0.4), (PUSH_STRENGTH, 0.3)],
        danger: 0.10,
    },
    Skill {
        train: &[(COMPRESSION, 0.6), (CORE_HOLD, 0.3), (HAMSTRING, 0.1)],
        rate: 0.26,
        prereq: &[(CORE_HOLD, 0.35), (HAMSTRING, 0.4)],
        danger: 0.05,
    },
    Skill {
        train: &[(PUSH_STRENGTH, 0.5), (HAMSTRING, 0.3), (COMPRESSION, 0.2)],
        rate: 0.24,
        prereq: &[(PUSH_STRENGTH, 0.4), (HAMSTRING, 0.5), (COMPRESSION, 0.5)],
        danger: 0.13,
    },
    Skill {
        train: &[(PUSH_STRENGTH, 0.5), (COMPRESSION, 0.3), (HAMSTRING, 0.15), (WRIST_FLEX, 0.05)],
        rate: 0.22,
        prereq: &[
            (PUSH_STRENGTH, 0.55),
            (COMPRESSION, 0.5),
            (HAMSTRING, 0.6),
            (WRIST_FLEX, 0.4),
            (SHOULDER_FLEX, 0.4),
            (BALANCE, 0.5),
        ],
        danger: 0.16,
    },
    Skill {
        train: &[
            (SHOULDER_FLEX, 0.15),
            (HAMSTRING, 0.15),
            (COMPRESSION, 0.15),
            (PUSH_STRENGTH, 0.15),
            (CORE_HOLD, 0.15),
            (BALANCE, 0.15),
            (WRIST_FLEX, 0.05),
            (SCRAP_BACK, 0.05),
        ],
        rate: 0.22,
        prereq: &[
            (PUSH_STRENGTH, 0.65),
            (COMPRESSION, 0.6),
            (HAMSTRING, 0.65),
            (WRIST_FLEX, 0.5),
            (SHOULDER_FLEX, 0.5),
            (CORE_HOLD, 0.55),
            (BALANCE, 0.55),
        ],
        danger: 0.20,
    },
];

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------
pub const WEEKS: usize = 36;
// Sum of weekly overload above this is treated as an injury event.
pub const INJURY_THRESHOLD: f32 = 0.16;
// Regression applied to the over-trained capacities on an injury.
pub const INJURY_STEP: f32 = 0.05;

/// Ordered-per-week focus distribution (softmax of network output).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

/// 0..1 "how ready is this athlete to train skill `s` right now" — the
/// gating factor. 1.0 = all prerequisites comfortably met; near 0 = not ready.
fn readiness(skill: &Skill, caps: &[f32; N_CAP]) -> f32 {
    if skill.prereq.is_empty() {
        return 1.0;
    }
    let mut worst = f32::INFINITY;
    for &(c, min) in skill.prereq {
        let ratio = (caps[c] / min).clamp(0.0, 1.0);
        if ratio < worst {
            worst = ratio;
        }
    }
    worst
}

/// Deterministic "raise the floor" algorithm: each week, concentrate focus on
/// the skill(s) that most efficiently train the currently-weakest capacities
/// the athlete is ready for. A simple, reliable scheduling baseline (no ML) —
/// evolution optimises toward this and beyond.
pub fn greedy_focus(caps: &[f32; N_CAP]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..N_CAP).collect();
    order.sort_by(|&a, &b| caps[a].partial_cmp(&caps[b]).unwrap());

    let mut focus = vec![0.0f32; SKILL_COUNT];
    for target_rank in 0..3 {
        let target_cap = order[target_rank];
        let mut best_score: f32 = 0.0;
        let mut best_s = 0usize;
        for (s, skill) in SKILLS.iter().enumerate() {
            let w: f32 = skill
                .train
                .iter()
                .filter(|(c, _)| *c == target_cap)
                .map(|(_, w)| w)
                .sum();
            if w <= 0.0 {
                continue;
            }
            let ready = readiness(skill, caps);
            let score = skill.rate * w * (1.0 - caps[target_cap]) * ready;
            if score > best_score {
                best_score = score;
                best_s = s;
            }
        }
        focus[best_s] += if target_rank == 0 { 0.55 } else if target_rank == 1 { 0.28 } else { 0.17 };
    }
    let sum: f32 = focus.iter().sum();
    if sum <= 0.0 {
        focus[0] = 1.0;
        return focus;
    }
    for f in focus.iter_mut() {
        *f /= sum;
    }
    focus
}

/// Attainment ("press capability" 0..1) combines the geometric mean (balanced
/// development) with a strong penalty for the *weakest* capacity. A press to
/// handstand is only possible when every prerequisite is dialed in, so one
/// lagging attribute caps the whole score — this is what drives the coach to
/// "raise the floor" rather than chase a favourite strength.
pub fn attainment(caps: &[f32; N_CAP]) -> f32 {
    let mut prod = 1.0f32;
    let mut min = 1.0f32;
    for &c in caps.iter() {
        prod *= c.max(1e-4);
        min = min.min(c);
    }
    let geo = prod.powf(1.0 / N_CAP as f32);
    // Blend: 38% geometric mean + 62% weakest link. The weak link dominates
    // because a press is only achievable when NO attribute is left behind.
    0.38 * geo + 0.62 * min
}

pub struct SimResult {
    /// Cumulative (and final) attainment over the horizon.
    pub final_score: f32,
    /// Number of injury events.
    pub injuries: u32,
    /// Weekly focus / attainment trace (for the UI pathway view).
    pub trace: Vec<f32>,
    /// Per-week dominant skill focus (index). Filled only when requested.
    pub plan: Vec<usize>,
}

/// Run one athlete through `weeks` weeks of training guided by the focus
/// distribution emitted each week from the coach net.
///
/// `target_plan` (when Some) records the argmax skill per week so the UI can
/// show an actual pathway.
pub fn simulate(
    start: &[f32; N_CAP],
    mut focus_fn: impl FnMut(&[f32; N_CAP]) -> Vec<f32>,
    weeks: usize,
    target_plan: bool,
) -> SimResult {
    let mut caps = *start;
    let mut injuries = 0u32;
    let mut trace = Vec::with_capacity(weeks);
    let mut plan = Vec::new();

    for _ in 0..weeks {
        let focus = focus_fn(&caps);
        let mut gain = [0.0f32; N_CAP];
        let mut overload = 0.0f32;
        // Which capacities are being pushed hardest (for injury attribution).
        let mut push = [0.0f32; N_CAP];

        for (s, &f) in focus.iter().enumerate() {
            if f <= 1e-4 {
                continue;
            }
            let skill = &SKILLS[s];
            let ready = readiness(skill, &caps);
            overload += f * (1.0 - ready) * skill.danger;
            for &(c, w) in skill.train {
                // Diminishing returns: gains shrink as the capacity grows.
                let g = f * skill.rate * w * (1.0 - caps[c]) * ready;
                gain[c] += g;
                push[c] += f * w;
            }
        }

        let overload_clamped = overload.clamp(0.0, 1.0);
        let recover = 1.0 - overload_clamped * 0.5;

        // Apply recovered gains.
        for c in 0..N_CAP {
            caps[c] = (caps[c] + gain[c] * recover).clamp(0.0, 1.0);
        }

        // Injury event.
        if overload > INJURY_THRESHOLD {
            injuries += 1;
            // Regress the capacities that were most heavily trained.
            let max_push = push.iter().cloned().fold(f32::MIN, f32::max).max(1e-6);
            for c in 0..N_CAP {
                if push[c] > 0.0 {
                    let frac = (push[c] / max_push).clamp(0.0, 1.0);
                    caps[c] = (caps[c] - INJURY_STEP * frac).clamp(0.0, 1.0);
                }
            }
            // Recovery week: mild decay across the board to punish overreach.
            for c in 0..N_CAP {
                caps[c] = (caps[c] - 0.01).clamp(0.0, 1.0);
            }
        }

        let at = attainment(&caps);
        trace.push(at);
        if target_plan {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in focus.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            plan.push(best);
        }
    }

    let final_score = attainment(&caps);
    SimResult {
        final_score,
        injuries,
        trace,
        plan,
    }
}
