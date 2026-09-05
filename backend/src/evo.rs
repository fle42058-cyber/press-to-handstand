//! Evolution orchestrator. Owns the population, evaluates individuals against
//! a fixed set of synthetic athlete *profiles*, tracks the best-of-generation
//! score (fitness), and exposes the best genome so far.

use crate::ga::{self, Params};
use crate::model;
use crate::nn::Net;
use crate::rng::Rng;

/// Network input = the 8 capacities; output = focus logits over the 10 skills.
/// Deliberately small so the genetic algorithm can search it effectively.
pub const SIZES: [usize; 3] = [8, 24, model::SKILL_COUNT];
/// Attainment considered "press to handstand achieved" (used for the reward).
pub const PRESS_THRESHOLD: f32 = 0.80;

/// How many synthetic athlete starting-profiles each individual is evaluated
/// against (so the evolved coach generalises, not overfits one body).
pub const N_PROFILES: usize = 10;

pub struct Evolution {
    pub sizes: Vec<usize>,
    params: Params,
    pub population: Vec<Net>,
    pub fitness: Vec<f32>,
    pub gen: u64,
    pub history: Vec<f32>, // best fitness per generation (append-only)
    pub best_genome: Vec<f32>,
    pub best_fitness: f32,
    pub profiles: Vec<[f32; model::N_CAP]>,
    rng: Rng,
}

impl Evolution {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let sizes = SIZES.to_vec();

        // Build a diverse spread of synthetic athlete profiles (deterministic),
        // plus a few representative "typical" athletes so the evolved coach is
        // directly good for a normal user like the demo defaults.
        let mut profiles = Vec::with_capacity(N_PROFILES);
        // Representative balanced athletes.
        for f in [0.25f32, 0.35, 0.45, 0.55, 0.65] {
            profiles.push([f; model::N_CAP]);
        }
        // A spread of varied / unbalanced athletes.
        while profiles.len() < N_PROFILES {
            let base = 0.12 + (profiles.len() as f32 / N_PROFILES as f32) * 0.30;
            let mut caps = [0.0f32; model::N_CAP];
            for v in caps.iter_mut() {
                *v = (base + rng.range(-0.14, 0.22)).clamp(0.02, 0.78);
            }
            profiles.push(caps);
        }

        let params = Params::default();
    // Seed element 0 with a hand-built heuristic coach (a "greedy: raise
    // the floor" policy) so the population always has a sensible baseline
    // and evolution can only improve on it. The rest are random starts.
    let mut population: Vec<Net> = (0..params.pop)
        .map(|_| Net::new_random(&sizes, &mut rng))
        .collect();
    population[0] = Net::from_genome(&sizes, heuristic_genome(&sizes, false));
        let fitness = vec![0.0f32; params.pop];
        let best_genome = population[0].genome.clone();

        Evolution {
            sizes,
            params,
            population,
            fitness,
            gen: 0,
            history: Vec::new(),
            best_genome,
            best_fitness: f32::NEG_INFINITY,
            profiles,
            rng,
        }
    }

    /// Evaluate the whole population and advance one generation.
    pub fn step(&mut self) {
        // Evaluate.
        for i in 0..self.population.len() {
            self.fitness[i] = evaluate_net(&self.profiles, &self.population[i]);
        }
        let best_idx = self
            .fitness
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if self.fitness[best_idx] > self.best_fitness {
            self.best_fitness = self.fitness[best_idx];
            self.best_genome = self.population[best_idx].genome.clone();
        }
        self.history.push(self.fitness[best_idx]);
        self.gen += 1;

        // Breed the next generation.
        self.population = ga::next_generation(
            &self.population,
            &self.fitness,
            &self.sizes,
            &self.params,
            &mut self.rng,
        );
    }

    /// The best-evolved coach so far.
    pub fn best_net(&self) -> Net {
        Net::from_genome(&self.sizes, self.best_genome.clone())
    }
}

/// Build a heuristic "raise the floor" coach by hand as a seed genome.
///
/// Each output logit scores a skill by how much it helps the currently-weak
/// capacities it trains (weighted by its learning rate) and how ready it is
/// (positive gating on its prerequisite capacities). This reproduces the
/// greedy baseline behaviour, so the GA starts from a sensible curriculum and
/// only refines it.
pub fn heuristic_genome(sizes: &[usize], tuned: bool) -> Vec<f32> {
    let in0 = sizes[0];
    let h = sizes[1];
    let out = sizes[2];
    let mut g = vec![0.0f32; Net::len_for(sizes)];

    // Hidden layer: unit j (< in0) passes capacity j through (identity);
    // remaining units stay near zero (act as a small bias reserve).
    let w0 = in0 * h;
    for j in 0..h {
        for i in 0..in0 {
            g[j * in0 + i] = if j == i { 1.0 } else { 0.0 };
        }
        g[w0 + j] = 0.0;
    }

    // Output layer.
    let w1 = w0 + h; // where output weights begin
    let (kappa, gamma, a, b, scale) = if tuned {
        (2.6, 5.0, 3.0, 0.2, 1.6)
    } else {
        (3.2, 1.6, 2.0, 0.12, 1.0)
    };
    for (s, skill) in model::SKILLS.iter().enumerate() {
        let base = w1 + s * h;
        for &(c, w) in skill.train {
            if c < in0 {
                g[base + c] += -kappa * skill.rate * w;
            }
        }
        for &(p, _min) in skill.prereq {
            if p < in0 {
                g[base + p] += gamma * skill.rate;
            }
        }
        // Preference for earlier, safer foundation skills.
        let bias_idx = w1 + out * h + s;
        g[bias_idx] = (a - (s as f32) * b) * scale;
        // scale output layer weights for sharper softmax concentration
        for j in 0..h {
            g[base + j] *= scale;
        }
    }
    g
}

/// Fitness of one genome: run it as the coach over all synthetic profiles
/// and reward end attainment + progression, minus injury cost.
fn evaluate_net(profiles: &[[f32; model::N_CAP]], net: &Net) -> f32 {
    let mut total = 0.0f32;
    for caps in profiles {
        let start_at = model::attainment(caps);
        let result = model::simulate(
            caps,
            |c| {
                let logits = net.forward(c);
                model::softmax(&logits)
            },
            model::WEEKS,
            true,
        );
        // Composite objective:
        //   +  strong weight on final press attainment
        //   +  reward steady progression (how much it climbed from baseline)
        //   +  big bonus for actually REACHING the press, scaled by how early
        //   -  heavy penalty per injury (safety)
        let reached = result.trace.iter().position(|&x| x >= PRESS_THRESHOLD);
        let mut score = result.final_score * 100.0;
        score += (result.final_score - start_at) * 40.0; // net progression
        if let Some(w) = reached {
            // earlier = better: 60 down to ~20 as the week approaches the end.
            let early = 1.0 - w as f32 / model::WEEKS as f32;
            score += 60.0 * (0.3 + 0.7 * early);
        }
        score -= result.injuries as f32 * 18.0;
        total += score;
    }
    total / profiles.len() as f32
}
