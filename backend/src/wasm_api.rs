#![allow(static_mut_refs)]
//! Browser bindings (compiled to `wasm32-unknown-unknown`).
//!
//! GitHub Pages cannot run a server, so this module exports plain
//! C-ABI functions over the same engine. The TypeScript client instantiates
//! this `.wasm` and drives the evolution + recommendations entirely in the
//! browser. State is single-threaded, so we use `static mut` globals.

use crate::evo::{self, Evolution};
use crate::model;
use crate::nn::Net;

const PRESS_THRESHOLD: f32 = evo::PRESS_THRESHOLD;
const PLAN_WEEKS: usize = model::WEEKS;

// Single-threaded store (wasm is single-threaded; the native binary never
// calls these helpers, but the module must compile for the rlib too).
static mut EVO: Option<Evolution> = None;
static mut OUT: Vec<u8> = Vec::new();
static mut SCRATCH: Vec<f32> = Vec::new();

fn get_evo_mut() -> &'static mut Evolution {
    unsafe {
        if EVO.is_none() {
            EVO = Some(Evolution::new(1337));
        }
        EVO.as_mut().unwrap()
    }
}
fn set_out(s: String) {
    unsafe {
        OUT = s.into_bytes();
    }
}
fn out_buf() -> &'static Vec<u8> {
    unsafe { &OUT }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            _ => o.push(c),
        }
    }
    o
}
fn floats_json(v: &[f32]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
    format!("[{}]", items.join(","))
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
/// (Re)initialise the evolution with a seed.
#[no_mangle]
pub extern "C" fn p2h_init(seed: u64) {
    unsafe { EVO = Some(Evolution::new(seed)) };
}
/// Advance one generation.
#[no_mangle]
pub extern "C" fn p2h_step() {
    get_evo_mut().step();
}
/// Advance `n` generations quickly (used to warm the population on load).
#[no_mangle]
pub extern "C" fn p2h_run(n: u64) {
    let evo = get_evo_mut();
    for _ in 0..n {
        evo.step();
    }
}
/// Pointer to the latest output buffer (set by meta/state/recommend).
#[no_mangle]
pub extern "C" fn p2h_out_ptr() -> *const u8 {
    out_buf().as_ptr()
}
/// Length of the latest output buffer.
#[no_mangle]
pub extern "C" fn p2h_out_len() -> usize {
    out_buf().len()
}
/// Fill a scratch buffer with `n` floats, return its pointer (to pass caps in).
#[no_mangle]
pub extern "C" fn p2h_alloc_f32(n: usize) -> *const f32 {
    unsafe {
        SCRATCH = vec![0.0f32; n];
        SCRATCH.as_ptr()
    }
}

/// Static model metadata (capacities, skills, horizon, threshold, goal).
#[no_mangle]
pub extern "C" fn p2h_meta() -> *const u8 {
    let caps: Vec<String> = model::CAPACITY_NAMES
        .iter()
        .zip(model::CAPACITY_HINT.iter())
        .map(|(n, h)| format!(r#"{{"name":"{}","hint":"{}"}}"#, esc(n), esc(h)))
        .collect();
    let skills: Vec<String> = model::SKILL_NAMES.iter().map(|n| format!(r#"{{"name":"{}"}}"#, esc(n))).collect();
    let s = format!(
        r#"{{"capacities":[{caps}],"skills":[{skills}],"weeks":{w},"threshold":{t},"goal":{g},"goalName":"{goal}"}}"#,
        caps = caps.join(","),
        skills = skills.join(","),
        w = model::WEEKS,
        t = PRESS_THRESHOLD,
        g = model::SKILL_COUNT - 1,
        goal = esc(model::SKILL_NAMES[model::SKILL_COUNT - 1]),
    );
    set_out(s);
    out_buf().as_ptr()
}

/// Evolution state JSON (gen, best fitness, population, downsampled history).
#[no_mangle]
pub extern "C" fn p2h_state() -> *const u8 {
    let evo = get_evo_mut();
    let step = (evo.history.len() / 300).max(1);
    let sampled: Vec<f32> = evo.history.iter().step_by(step).copied().collect();
    let s = format!(
        r#"{{"gen":{},"best_fitness":{:.3},"population":{},"running":true,"history":{}}}"#,
        evo.gen,
        evo.best_fitness,
        evo.population.len(),
        floats_json(&sampled),
    );
    set_out(s);
    out_buf().as_ptr()
}

/// Generate a recommended pathway given a pointer to 8 capacity floats.
#[no_mangle]
pub extern "C" fn p2h_recommend(ptr: *const f32, n: usize) -> *const u8 {
    let mut caps = [0.5f32; model::N_CAP];
    unsafe {
        let sl = std::slice::from_raw_parts(ptr, n.min(model::N_CAP));
        for (i, &v) in sl.iter().enumerate() {
            caps[i] = v.clamp(0.0, 1.0);
        }
    }

    let net: Net = get_evo_mut().best_net();
    let evolved = model::simulate(
        &caps,
        |c| {
            let logits = net.forward(c);
            model::softmax(&logits)
        },
        PLAN_WEEKS,
        true,
    );
    let baseline = model::simulate(&caps, |c| model::greedy_focus(c), PLAN_WEEKS, true);

    let evo_score = evolved.final_score;
    let base_score = baseline.final_score;
    let (result, source) = if evo_score >= base_score {
        (evolved, "evolved")
    } else {
        (baseline, "baseline")
    };

    // Blocks by dominant skill.
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    let mut idx = 0usize;
    while idx < result.plan.len() {
        let skill = result.plan[idx];
        let start = idx;
        let mut end = idx;
        while end + 1 < result.plan.len() && result.plan[end + 1] == skill {
            end += 1;
        }
        blocks.push((skill, start, end - start + 1));
        idx = end + 1;
    }
    let mut focus_sum = vec![0.0f32; model::SKILL_COUNT];
    for (skill, _, count) in &blocks {
        focus_sum[*skill] += *count as f32;
    }
    let mut ranking: Vec<usize> = (0..model::SKILL_COUNT).collect();
    ranking.sort_by(|&a, &b| focus_sum[b].partial_cmp(&focus_sum[a]).unwrap());

    let mut blocks_json = Vec::new();
    for &(skill, start, count) in &blocks {
        blocks_json.push(format!(
            r#"{{"skill":{},"name":"{}","weekStart":{},"weeks":{}}}"#,
            skill,
            esc(model::SKILL_NAMES[skill]),
            start,
            count
        ));
    }
    let mut rank_json = Vec::new();
    for (i, &s) in ranking.iter().enumerate().take(model::SKILL_COUNT) {
        rank_json.push(format!(
            r#"{{"rank":{},"skill":{},"name":"{}","focus":{:.3}}}"#,
            i,
            s,
            esc(model::SKILL_NAMES[s]),
            focus_sum[s]
        ));
    }

    let weeks_to_press = result
        .trace
        .iter()
        .position(|&t| t >= PRESS_THRESHOLD)
        .map(|p| p + 1)
        .unwrap_or(0);
    let start_at = model::attainment(&caps);
    let gen = get_evo_mut().gen;
    let s = format!(
        r#"{{"caps":{},"startAttainment":{:.3},"finalAttainment":{:.3},"weeksToPress":{},"injuries":{},"source":"{}","evoAttainment":{:.3},"baseAttainment":{:.3},"gen":{},"trace":{},"blocks":[{}],"ranking":[{}]}}"#,
        floats_json(&caps),
        start_at,
        result.final_score,
        weeks_to_press,
        result.injuries,
        source,
        evo_score,
        base_score,
        gen,
        floats_json(&result.trace),
        blocks_json.join(","),
        rank_json.join(","),
    );
    set_out(s);
    out_buf().as_ptr()
}
