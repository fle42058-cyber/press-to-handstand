//! A tiny fully-connected feed-forward network. Weights+biases are stored as a
//! single flat "genome" of f32 so the genetic algorithm can treat an
//! individual as a plain vector (crossover + mutation operate on the genome).

#[derive(Clone)]
pub struct Net {
    pub sizes: Vec<usize>,
    pub genome: Vec<f32>,
}

impl Net {
    /// Total number of genome slots for a given layer layout:
    /// for each layer pair: (in+1) * out  (the +1 is a bias per output node).
    pub fn len_for(sizes: &[usize]) -> usize {
        let mut n = 0;
        for w in 0..sizes.len() - 1 {
            n += (sizes[w] + 1) * sizes[w + 1];
        }
        n
    }

    pub fn new_random(sizes: &[usize], rng: &mut crate::rng::Rng) -> Self {
        let len = Self::len_for(sizes);
        // Small initial weights so outputs start near a broad, safe softmax.
        let genome = (0..len).map(|_| rng.range(-0.8, 0.8)).collect();
        Net {
            sizes: sizes.to_vec(),
            genome,
        }
    }

    pub fn from_genome(sizes: &[usize], genome: Vec<f32>) -> Self {
        Net {
            sizes: sizes.to_vec(),
            genome,
        }
    }

    /// Forward pass. Returns the raw logits of the output layer.
    /// Hidden layers use tanh; the output layer is left linear (the caller
    /// applies softmax so magnitudes control confidence).
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut act = input.to_vec();
        let mut idx = 0usize;
        let l = self.sizes.len();
        for w in 0..l - 1 {
            let in_s = self.sizes[w];
            let out_s = self.sizes[w + 1];
            let mut next = vec![0.0f32; out_s];
            for o in 0..out_s {
                let base = idx + o * in_s;
                let mut sum = 0.0f32;
                for (i, &a) in act.iter().enumerate() {
                    sum += a * self.genome[base + i];
                }
                // bias is stored after all the (in_s * out_s) weights for this layer
                sum += self.genome[idx + in_s * out_s + o];
                next[o] = if w == l - 2 { sum } else { sum.tanh() };
            }
            idx += in_s * out_s + out_s;
            act = next;
        }
        act
    }
}
