//! Deterministic, dependency-free noise: integer hashing, value noise,
//! Perlin-style gradient noise, and FBM combinators.
//!
//! Everything is a pure function of (position, seed) — identical output
//! across runs, threads, and machines with the same inputs.

use glam::{Vec2, Vec3};

/// Full-avalanche 32-bit integer hash (triple multiply-xorshift).
#[inline]
fn avalanche(mut x: u32) -> u32 {
    x ^= x >> 17;
    x = x.wrapping_mul(0xed5a_d4bb);
    x ^= x >> 11;
    x = x.wrapping_mul(0xac4c_1b51);
    x ^= x >> 15;
    x = x.wrapping_mul(0x3184_8bab);
    x ^= x >> 14;
    x
}

/// Hash a 2-D lattice point with a seed.
#[inline]
pub fn hash2(ix: i32, iy: i32, seed: u32) -> u32 {
    avalanche(
        (ix as u32)
            .wrapping_mul(0x8529_7a4d)
            .wrapping_add((iy as u32).wrapping_mul(0x68e3_1da4))
            ^ seed,
    )
}

/// Hash a 3-D lattice point with a seed.
#[inline]
pub fn hash3(ix: i32, iy: i32, iz: i32, seed: u32) -> u32 {
    avalanche(
        (ix as u32)
            .wrapping_mul(0x8529_7a4d)
            .wrapping_add((iy as u32).wrapping_mul(0x68e3_1da4))
            .wrapping_add((iz as u32).wrapping_mul(0x1b87_3593))
            ^ seed,
    )
}

/// Map a hash to [0, 1).
#[inline]
fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Quintic fade curve `6t⁵ - 15t⁴ + 10t³` (C² continuous at lattice points).
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// One of eight unit-ish gradient directions.
#[inline]
fn grad2(h: u32) -> Vec2 {
    const DIAG: f32 = std::f32::consts::FRAC_1_SQRT_2;
    match h & 7 {
        0 => Vec2::new(1.0, 0.0),
        1 => Vec2::new(-1.0, 0.0),
        2 => Vec2::new(0.0, 1.0),
        3 => Vec2::new(0.0, -1.0),
        4 => Vec2::new(DIAG, DIAG),
        5 => Vec2::new(-DIAG, DIAG),
        6 => Vec2::new(DIAG, -DIAG),
        _ => Vec2::new(-DIAG, -DIAG),
    }
}

/// 2-D Perlin-style gradient noise in ≈[-1, 1].
pub fn gradient2(p: Vec2, seed: u32) -> f32 {
    let cell = p.floor();
    let f = p - cell;
    let (ix, iy) = (cell.x as i32, cell.y as i32);

    let dot = |dx: i32, dy: i32| -> f32 {
        let g = grad2(hash2(ix + dx, iy + dy, seed));
        g.dot(f - Vec2::new(dx as f32, dy as f32))
    };
    let (u, v) = (fade(f.x), fade(f.y));
    let x0 = lerp(dot(0, 0), dot(1, 0), u);
    let x1 = lerp(dot(0, 1), dot(1, 1), u);
    // Theoretical range is ±√2/2 for unit gradients; rescale toward ±1.
    lerp(x0, x1, v) * std::f32::consts::SQRT_2
}

/// 3-D value noise in [-1, 1].
pub fn value3(p: Vec3, seed: u32) -> f32 {
    let cell = p.floor();
    let f = p - cell;
    let (ix, iy, iz) = (cell.x as i32, cell.y as i32, cell.z as i32);

    let val = |dx: i32, dy: i32, dz: i32| -> f32 {
        unit(hash3(ix + dx, iy + dy, iz + dz, seed)) * 2.0 - 1.0
    };
    let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
    let x00 = lerp(val(0, 0, 0), val(1, 0, 0), u);
    let x10 = lerp(val(0, 1, 0), val(1, 1, 0), u);
    let x01 = lerp(val(0, 0, 1), val(1, 0, 1), u);
    let x11 = lerp(val(0, 1, 1), val(1, 1, 1), u);
    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);
    lerp(y0, y1, w)
}

/// Fractal Brownian motion over 2-D gradient noise, normalized to ≈[-1, 1].
#[derive(Copy, Clone, Debug)]
pub struct Fbm {
    pub octaves: u8,
    pub lacunarity: f32,
    pub gain: f32,
    pub seed: u32,
}

impl Fbm {
    pub fn new(octaves: u8, seed: u32) -> Self {
        Self {
            octaves,
            lacunarity: 2.0,
            gain: 0.5,
            seed,
        }
    }

    /// Sample at `p` (feature wavelength ≈ 1 unit at the first octave).
    pub fn sample2(&self, p: Vec2) -> f32 {
        let mut sum = 0.0;
        let mut amp = 1.0;
        let mut freq = 1.0;
        let mut total = 0.0;
        for octave in 0..self.octaves {
            // Decorrelate octaves with distinct seeds.
            let seed = self
                .seed
                .wrapping_add(u32::from(octave).wrapping_mul(0x9e37_79b9));
            sum += gradient2(p * freq, seed) * amp;
            total += amp;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
        sum / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random sample points for the tests.
    fn sample_points(n: usize) -> Vec<Vec2> {
        (0..n)
            .map(|i| {
                let h1 = avalanche(i as u32 ^ 0xA5A5_5A5A);
                let h2 = avalanche(h1 ^ 0x0F0F_F0F0);
                Vec2::new(unit(h1) * 2000.0 - 1000.0, unit(h2) * 2000.0 - 1000.0)
            })
            .collect()
    }

    #[test]
    fn gradient_noise_is_deterministic() {
        for p in sample_points(500) {
            assert_eq!(gradient2(p, 42).to_bits(), gradient2(p, 42).to_bits());
        }
        let fbm = Fbm::new(5, 1337);
        for p in sample_points(200) {
            assert_eq!(fbm.sample2(p).to_bits(), fbm.sample2(p).to_bits());
        }
    }

    #[test]
    fn outputs_stay_in_range() {
        let fbm = Fbm::new(5, 7);
        for p in sample_points(10_000) {
            let g = gradient2(p, 7);
            assert!((-1.001..=1.001).contains(&g), "gradient2({p}) = {g}");
            let f = fbm.sample2(p * 0.13);
            assert!((-1.001..=1.001).contains(&f), "fbm({p}) = {f}");
            let v = value3(Vec3::new(p.x, p.y, p.x * 0.5), 7);
            assert!((-1.0..=1.0).contains(&v), "value3 = {v}");
        }
    }

    #[test]
    fn noise_is_continuous() {
        const EPS: f32 = 1e-3;
        for p in sample_points(1_000) {
            let base = gradient2(p, 99);
            for d in [Vec2::new(EPS, 0.0), Vec2::new(0.0, EPS)] {
                let step = gradient2(p + d, 99);
                assert!(
                    (base - step).abs() < 0.05,
                    "discontinuity at {p}: {base} vs {step}"
                );
            }
        }
    }

    #[test]
    fn different_seeds_produce_different_fields() {
        let mut differing = 0usize;
        let points = sample_points(1_000);
        for &p in &points {
            let a = gradient2(p, 1);
            let b = gradient2(p, 2);
            if (a - b).abs() > 1e-6 {
                differing += 1;
            }
        }
        assert!(
            differing > 900,
            "seeds 1 and 2 differ at only {differing}/1000 samples"
        );
    }

    #[test]
    fn value3_varies_along_each_axis() {
        // Guards against a hash that ignores one coordinate.
        let base = value3(Vec3::new(10.3, 20.7, 30.1), 5);
        assert!((value3(Vec3::new(11.3, 20.7, 30.1), 5) - base).abs() > 1e-6);
        assert!((value3(Vec3::new(10.3, 21.7, 30.1), 5) - base).abs() > 1e-6);
        assert!((value3(Vec3::new(10.3, 20.7, 31.1), 5) - base).abs() > 1e-6);
    }

    #[test]
    fn fbm_octaves_add_detail() {
        // More octaves must change the field (detail added), while staying
        // correlated with the base octave at large scales.
        let one = Fbm::new(1, 11);
        let five = Fbm::new(5, 11);
        let mut changed = 0usize;
        let points = sample_points(500);
        for &p in &points {
            if (one.sample2(p * 0.03) - five.sample2(p * 0.03)).abs() > 1e-4 {
                changed += 1;
            }
        }
        assert!(changed > 450, "octaves changed only {changed}/500 samples");
    }
}
