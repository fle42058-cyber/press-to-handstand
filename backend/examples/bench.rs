//! Quick convergence / sanity harness (run: cargo run --release --example bench)

use p2h_engine::evo::{Evolution, PRESS_THRESHOLD as PRESS, SIZES, heuristic_genome};
use p2h_engine::model;
use p2h_engine::nn::Net;

fn print_plan(r: &model::SimResult) {
    let mut idx = 0;
    while idx < r.plan.len() {
        let s = r.plan[idx];
        let mut end = idx;
        while end + 1 < r.plan.len() && r.plan[end + 1] == s {
            end += 1;
        }
        println!("weeks {}-{}: {}", idx + 1, end + 1, model::SKILL_NAMES[s]);
        idx = end + 1;
    }
}

/// Greedy "raise the floor": each week, concentrate focus on the single skill
/// that most efficiently trains the currently-weakest capacity that it is
/// ready for. A straightforward baseline the evolved net should beat.
fn greedy(caps: &[f32; model::N_CAP]) -> Vec<f32> {
    // expected per-unit-focus gain each skill gives to the weakest caps.
    // We pick the skill that maximizes the gain on the current weakest cap
    // (weighted by readiness), then give it most of the focus.
    let mut order: Vec<usize> = (0..model::N_CAP).collect();
    order.sort_by(|&a, &b| caps[a].partial_cmp(&caps[b]).unwrap());

    let mut focus = vec![0.0f32; model::SKILL_COUNT];
    for target_rank in 0..3 {
        let target_cap = order[target_rank];
        let mut best_score: f32 = 0.0;
        let mut best_s = 0usize;
        for (s, skill) in model::SKILLS.iter().enumerate() {
            let w: f32 = skill
                .train
                .iter()
                .filter(|(c, _)| *c == target_cap)
                .map(|(_, w)| w)
                .sum();
            if w <= 0.0 {
                continue;
            }
            // readiness gate
            let ready = if skill.prereq.is_empty() {
                1.0
            } else {
                let mut worst = 1.0f32;
                for &(c, mn) in skill.prereq {
                    worst = worst.min((caps[c] / mn).clamp(0.0, 1.0));
                }
                worst
            };
            let score = skill.rate * w * (1.0 - caps[target_cap]) * ready;
            if score > best_score {
                best_score = score;
                best_s = s;
            }
        }
        // put focus on the best skill (diminished by rank: 0.55, 0.28, 0.17)
        focus[best_s] += if target_rank == 0 { 0.55 } else if target_rank == 1 { 0.28 } else { 0.17 };
    }
    let ssum: f32 = focus.iter().sum();
    if ssum <= 0.0 {
        focus[0] = 1.0;
        return focus;
    }
    focus.iter().map(|x| x / ssum).collect()
}

fn main() {
    let mut ev = Evolution::new(1337);
    let gens: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);

    println!("gen  best_fitness   worst_final  best_final");
    let num = 20;
    for g in 0..gens {
        ev.step();
        if g % num == 0 || g == gens - 1 {
            let net = ev.best_net();
            let mut worst = 1.0f32;
            let mut bestline: f32 = 0.0;
            let mut avg = 0.0f32;
            for caps in &ev.profiles {
                let r = model::simulate(caps, |c| model::softmax(&net.forward(c)), model::WEEKS, true);
                worst = worst.min(r.final_score);
                bestline = bestline.max(r.final_score);
                avg += r.final_score;
            }
            avg /= ev.profiles.len() as f32;
            println!("{:4}  {:10.2}   {:.3}  {:.3}  avg {:.3}", ev.gen, ev.best_fitness, worst, bestline, avg);
        }
    }

    // How often does the evolved coach reach the press across all profiles?
    let net = ev.best_net();
    let mut reached = 0;
    let mut finals = vec![];
    for caps in &ev.profiles {
        let r = model::simulate(caps, |c| model::softmax(&net.forward(c)), model::WEEKS, true);
        finals.push(r.final_score);
        if r.trace.iter().any(|&x| x >= PRESS) {
            reached += 1;
        }
    }
    finals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "reached press: {}/{} profiles; final range {:.3}-{:.3}",
        reached,
        ev.profiles.len(),
        finals[0],
        finals[finals.len() - 1]
    );

    let caps = [0.35f32; model::N_CAP];

    println!("\n-- heuristic seed (tuned) on mid-level athlete --");
    {
        let net = Net::from_genome(&SIZES, heuristic_genome(&SIZES, true));
        let r = model::simulate(&caps, |c| model::softmax(&net.forward(c)), model::WEEKS, true);
        println!("  seed final {:.3}, weeks to 0.80: {:?}", r.final_score,
            r.trace.iter().position(|&x| x >= PRESS).map(|p| p + 1));
    }

    println!("\n-- evolved plan (mid-level athlete) --");
    let net = ev.best_net();
    let r = model::simulate(&caps, |c| model::softmax(&net.forward(c)), model::WEEKS, true);
    print_plan(&r);
    println!("final: {:.3}, weeks to 0.80: {:?}", r.final_score,
        r.trace.iter().position(|&x| x >= 0.80).map(|p| p + 1));

    println!("\n-- greedy baseline (mid-level athlete) --");
    let r2 = model::simulate(&caps, greedy, model::WEEKS, true);
    print_plan(&r2);
    println!("final: {:.3}, weeks to 0.80: {:?}", r2.final_score,
        r2.trace.iter().position(|&x| x >= 0.80).map(|p| p + 1));
}
