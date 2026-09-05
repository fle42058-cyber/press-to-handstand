//! Genetic algorithm operating on flat genomes (the neural-net weight vectors).
//! Pure classical evolution: ranking + tournament selection, uniform
//! crossover, Gaussian mutation, elitism. No ML libraries.

use crate::nn::Net;
use crate::rng::Rng;

pub struct Params {
    pub pop: usize,
    pub elite: usize,        // top-N carried unchanged
    pub tournament: usize,   // tournament size
    pub crossover_rate: f32, // probability of crossover per child
    pub mutation_rate: f32,  // probability per genome slot
    pub mutation_sigma: f32, // gaussian sigma
}

impl Default for Params {
    fn default() -> Self {
        Params {
            pop: 64,
            elite: 2,
            tournament: 4,
            crossover_rate: 0.9,
            mutation_rate: 0.30,
            mutation_sigma: 0.45,
        }
    }
}

/// Select one parent via tournament selection over `fitness`.
fn select(parents: &[Net], fitness: &[f32], k: usize, rng: &mut Rng) -> Net {
    let mut best_idx = rng.below(parents.len());
    for _ in 1..k {
        let j = rng.below(parents.len());
        if fitness[j] > fitness[best_idx] {
            best_idx = j;
        }
    }
    parents[best_idx].clone()
}

/// Breed one offspring from the population.
fn breed(
    parents: &[Net],
    fitness: &[f32],
    sizes: &[usize],
    p: &Params,
    rng: &mut Rng,
) -> Vec<f32> {
    let a = select(parents, fitness, p.tournament, rng);
    let b = select(parents, fitness, p.tournament, rng);
    let mut child = Vec::with_capacity(a.genome.len());

    for i in 0..a.genome.len() {
        // Uniform crossover.
        let g = if rng.next_f() < p.crossover_rate {
            if rng.next_f() < 0.5 {
                a.genome[i]
            } else {
                b.genome[i]
            }
        } else {
            a.genome[i]
        };
        // Gaussian mutation.
        let mut g = g;
        if rng.next_f() < p.mutation_rate {
            g += rng.gauss() * p.mutation_sigma;
            // keep weights bounded to avoid blow-up
            g = g.clamp(-6.0, 6.0);
        }
        child.push(g);
    }

    // Occasionally do a fresh random mutation burst to keep diversity — an
    // escape hatch for a plateaued population.
    if rng.next_f() < 0.08 {
        for g in child.iter_mut() {
            *g = rng.range(-0.8, 0.8);
        }
    }

    let _ = sizes; // reserved for future structural mutation
    child
}

/// Produce the next generation given current fitness scores.
pub fn next_generation(
    population: &[Net],
    fitness: &[f32],
    sizes: &[usize],
    p: &Params,
    rng: &mut Rng,
) -> Vec<Net> {
    // Rank by fitness (descending); order indices.
    let mut order: Vec<usize> = (0..population.len()).collect();
    order.sort_by(|&i, &j| fitness[j].partial_cmp(&fitness[i]).unwrap());

    let mut next = Vec::with_capacity(population.len());

    // Elitism: carry the best unchanged.
    for &i in order.iter().take(p.elite) {
        next.push(population[i].clone());
    }

    // Fill the rest with offspring.
    while next.len() < population.len() {
        let genome = breed(population, fitness, sizes, p, rng);
        next.push(Net::from_genome(sizes, genome));
    }

    next
}
