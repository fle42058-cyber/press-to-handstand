//! Minimal deterministic PRNG (xorshift64*) so the engine needs no crates.

#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid zero state.
        let state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Rng { state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn next_f(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [a, b).
    #[inline]
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + self.next_f() * (b - a)
    }

    /// Standard normal via Box-Muller.
    pub fn gauss(&mut self) -> f32 {
        let u1 = (self.next_f() * 2.0 - 1.0).max(1e-9);
        let u2 = self.next_f();
        let r = (-2.0 * u1.abs().ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}
