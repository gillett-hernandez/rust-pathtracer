use crate::math::*;
use crate::renderer::Film;

// use std::error::Error;
// use std::fs::File;
// use std::io::Read;

use math::spectral::Op;
use packed_simd::f32x4;

#[derive(Clone)]
pub struct Texture4 {
    pub curves: [CDF; 4],
    pub texture: Film<f32x4>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture4 {
    // evaluate the 4 CDFs with the mixing ratios specified by the texture.
    // not clamped to 0 to 1, so that should be done by the callee
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factors = self.texture.at_uv(uv);
        let eval = f32x4::new(
            self.curves[0].evaluate_power(lambda),
            self.curves[1].evaluate_power(lambda),
            self.curves[2].evaluate_power(lambda),
            self.curves[3].evaluate_power(lambda),
        );
        (factors * eval).sum()
    }

    pub fn curve_at(&self, uv: (f32, f32)) -> SPD {
        let texel = self.texture.at_uv(uv);
        SPD::Machine {
            list: vec![
                (
                    Op::Add,
                    SPD::Machine {
                        list: vec![(Op::Mul, self.curves[0].pdf)],
                        seed: texel.extract(0),
                    },
                ),
                (
                    Op::Add,
                    SPD::Machine {
                        list: vec![(Op::Mul, self.curves[1].pdf)],
                        seed: texel.extract(1),
                    },
                ),
                (
                    Op::Add,
                    SPD::Machine {
                        list: vec![(Op::Mul, self.curves[2].pdf)],
                        seed: texel.extract(2),
                    },
                ),
                (
                    Op::Add,
                    SPD::Machine {
                        list: vec![(Op::Mul, self.curves[3].pdf)],
                        seed: texel.extract(3),
                    },
                ),
            ],
            seed: 0.0,
        }
    }
}
#[derive(Clone)]
pub struct Texture1 {
    pub curve: CDF,
    pub texture: Film<f32>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture1 {
    // evaluate the 4 CDFs with the mixing ratios specified by the texture.
    // not clamped to 0 to 1, so that should be done by the callee
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factor = self.texture.at_uv(uv);
        let eval = self.curve.evaluate_power(lambda);
        factor * eval
    }

    pub fn curve_at(&self, uv: (f32, f32)) -> SPD {
        SPD::Machine {
            list: vec![(Op::Mul, self.curve.pdf)],
            seed: self.texture.at_uv(uv),
        }
    }
}
#[derive(Clone)]
pub enum Texture {
    Texture1(Texture1),
    Texture4(Texture4),
}

impl Texture {
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        match self {
            Texture::Texture1(tex) => tex.eval_at(lambda, uv),
            Texture::Texture4(tex) => tex.eval_at(lambda, uv),
        }
    }

    // pub fn curve_at(&self, uv: (f32, f32)) -> CDF {}
}

#[derive(Clone)]
pub struct TexStack {
    pub textures: Vec<Texture>,
}
impl TexStack {
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        let mut energy = 0.0;
        for tex in self.textures.iter() {
            energy += tex.eval_at(lambda, uv);
        }
        energy
    }
    pub fn importance_sample_at(
        &self,
        uv: (f32, f32),
        sample: Sample1D,
    ) -> (SingleWavelength, PDF) {
        let mut spds: Vec<SPD> = Vec::new();
        let mut cumulative_integral = 0.0;
        let mut s = 0.0;
        for spd in &self.textures {}
        unimplemented!()
    }
    pub fn bake_importance_map(&self, width: usize, height: usize) -> Film<f32> {
        let mut film = Film::new(width, height, 0.0f32);
        let mut line_luminance = 0.0;
        let mut cumulative_luminance = 0.0;
        for y in 0..height {
            for x in 0..width {
                let uv = (x as f32 / width as f32, y as f32 / height as f32);
                let mut luminance = 0.0;
                for tex in &self.textures {
                    luminance += match tex {
                        Texture::Texture1(inner) => {
                            inner.curve.cdf_integral * inner.texture.at_uv(uv)
                        }
                        Texture::Texture4(inner) => (f32x4::new(
                            inner.curves[0].cdf_integral,
                            inner.curves[1].cdf_integral,
                            inner.curves[2].cdf_integral,
                            inner.curves[3].cdf_integral,
                        ) * inner.texture.at_uv(uv))
                        .sum(),
                    };
                }
                film.buffer[y * width + x] = luminance;
            }
        }
        film
    }
}
