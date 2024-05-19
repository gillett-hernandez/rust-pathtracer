use crate::prelude::*;

use math::curves::{Curve, CurveWithCDF, InterpolationMode, Op};

// Into

// trait MyInto<T>: Sized {
//     fn force_into(self) -> T;
// }

// impl MyInto<f32x4> for f32 {
//     fn force_into(self) -> f32x4 {
//         f32x4::splat(self)
//     }
// }

// impl MyInto<f32> for f32 {
//     // trivially
//     fn force_into(self) -> f32 {
//         self
//     }
// }

pub trait EvalAt<T: Field>
where
    CurveWithCDF: SpectralPowerDistributionFunction<T>,
{
    fn eval_at(&self, lambda: T, uv: (f32, f32)) -> T;
}

#[derive(Clone)]
pub struct Texture4 {
    pub curves: [CurveWithCDF; 4],
    pub texture: Vec2D<f32x4>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture4 {
    // evaluate the 4 CDFs with the mixing ratios specified by the texture.
    // not clamped to 0 to 1, so that should be done by the callee

    pub fn curve_at(&self, uv: (f32, f32)) -> Curve {
        let texel = self.texture.at_uv(uv);
        Curve::Machine {
            list: vec![
                (
                    Op::Add,
                    Curve::Machine {
                        list: vec![(Op::Mul, self.curves[0].pdf.clone())],
                        seed: texel[0],
                    },
                ),
                (
                    Op::Add,
                    Curve::Machine {
                        list: vec![(Op::Mul, self.curves[1].pdf.clone())],
                        seed: texel[1],
                    },
                ),
                (
                    Op::Add,
                    Curve::Machine {
                        list: vec![(Op::Mul, self.curves[2].pdf.clone())],
                        seed: texel[2],
                    },
                ),
                (
                    Op::Add,
                    Curve::Machine {
                        list: vec![(Op::Mul, self.curves[3].pdf.clone())],
                        seed: texel[3],
                    },
                ),
            ],
            seed: 0.0,
        }
    }
}

// evaluate the 4 CDFs with the mixing ratios specified by the texture.
// not clamped to 0 to 1, so that should be done by the callee
impl EvalAt<f32x4> for Texture4 {
    fn eval_at(&self, lambda: f32x4, uv: (f32, f32)) -> f32x4 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let [x, y, z, w]: [f32; 4] = self.texture.at_uv(uv).into();
        // let eval = f32x4::new(
        let evals = [
            self.curves[0].evaluate_power(lambda),
            self.curves[1].evaluate_power(lambda),
            self.curves[2].evaluate_power(lambda),
            self.curves[3].evaluate_power(lambda),
        ];

        evals[0] * f32x4::splat(x)
            + evals[1] * f32x4::splat(y)
            + evals[2] * f32x4::splat(z)
            + evals[3] * f32x4::splat(w)
    }
}

impl EvalAt<f32> for Texture4 {
    #[inline(always)]
    fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let texel = self.texture.at_uv(uv);
        // let eval = f32x4::new(
        let evals = f32x4::from_array([
            self.curves[0].evaluate_power(lambda),
            self.curves[1].evaluate_power(lambda),
            self.curves[2].evaluate_power(lambda),
            self.curves[3].evaluate_power(lambda),
        ]);

        (evals * texel).reduce_sum()
    }
}

#[derive(Clone)]
pub struct Texture1 {
    pub curve: CurveWithCDF,
    pub texture: Vec2D<f32>,
    pub interpolation_mode: InterpolationMode,
}

impl Texture1 {
    pub fn curve_at(&self, uv: (f32, f32)) -> Curve {
        Curve::Machine {
            list: vec![(Op::Mul, self.curve.pdf.clone())],
            seed: self.texture.at_uv(uv),
        }
    }
}

impl EvalAt<f32> for Texture1 {
    #[inline(always)]
    fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factor = self.texture.at_uv(uv);
        let eval = self.curve.evaluate_power(lambda);
        eval * factor
    }
}

impl EvalAt<f32x4> for Texture1 {
    #[inline(always)]
    fn eval_at(&self, lambda: f32x4, uv: (f32, f32)) -> f32x4 {
        // TODO: bilinear or bicubic texture interpolation/filtering
        let factor = self.texture.at_uv(uv);
        let eval = self.curve.evaluate_power(lambda);
        eval * f32x4::splat(factor)
    }
}

#[derive(Clone)]
pub enum Texture {
    Texture1(Texture1),
    Texture4(Texture4),
}

impl Texture {
    pub fn curve_at(&self, uv: (f32, f32)) -> Curve {
        match self {
            Texture::Texture1(tex) => tex.curve_at(uv),
            Texture::Texture4(tex) => tex.curve_at(uv),
        }
    }
}

// TODO: bilinear or bicubic texture interpolation/filtering
impl EvalAt<f32> for Texture {
    fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        match self {
            Texture::Texture1(tex) => tex.eval_at(lambda, uv),
            Texture::Texture4(tex) => tex.eval_at(lambda, uv),
        }
    }
}

// TODO: bilinear or bicubic texture interpolation/filtering
impl EvalAt<f32x4> for Texture {
    fn eval_at(&self, lambda: f32x4, uv: (f32, f32)) -> f32x4 {
        match self {
            Texture::Texture1(tex) => tex.eval_at(lambda, uv),
            Texture::Texture4(tex) => tex.eval_at(lambda, uv),
        }
    }
}

pub fn replace_channel(tex4: &mut Texture4, tex1: Texture1, channel: u8) {
    match channel {
        4.. => {
            // technically reachable but
            panic!("bad channel for call to replace_channel. should be less than 4");
        }
        _ => {}
    }
    let channel = channel as usize;

    tex4.texture
        .buffer
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, pixel)| {
            pixel[3] = tex1.texture.buffer[idx];
        });
    tex4.curves[channel] = tex1.curve;
}

#[derive(Clone)]
pub struct TexStack {
    pub textures: Vec<Texture>,
}

impl TexStack {
    // pub fn importance_sample_at(
    //     &self,
    //     uv: (f32, f32),
    //     sample: Sample1D,
    // ) -> (SingleWavelength, PDF) {
    //     // let mut spds: Vec<Curve> = Vec::new();
    //     // let mut cumulative_integral = 0.0;
    //     // let mut s = 0.0;
    //     // for spd in &self.textures {}
    //     todo!()
    // }

    pub fn curve_at(&self, uv: (f32, f32)) -> Curve {
        let mut list = Vec::new();
        let seed = 0.0;
        for tex in &self.textures {
            list.push((Op::Add, tex.curve_at(uv)));
        }
        Curve::Machine { seed, list }
    }
    // pub fn bake_importance_map(&self, width: usize, height: usize) -> Film<f32> {
    //     let mut film = Film::new(width, height, 0.0f32);
    //     let mut line_luminance = 0.0;
    //     let mut cumulative_luminance = 0.0;
    //     for y in 0..height {
    //         for x in 0..width {
    //             let uv = (x as f32 / width as f32, y as f32 / height as f32);
    //             let mut luminance = 0.0;
    //             for tex in &self.textures {
    //                 luminance += match tex {
    //                     Texture::Texture1(inner) => {
    //                         inner.curve.cdf_integral * inner.texture.at_uv(uv)
    //                     }
    //                     Texture::Texture4(inner) => (f32x4::new(
    //                         inner.curves[0].cdf_integral,
    //                         inner.curves[1].cdf_integral,
    //                         inner.curves[2].cdf_integral,
    //                         inner.curves[3].cdf_integral,
    //                     ) * inner.texture.at_uv(uv))
    //                     .sum(),
    //                 };
    //             }
    //             film.buffer[y * width + x] = luminance;
    //         }
    //     }
    //     film
    // }
}

impl EvalAt<f32> for TexStack {
    fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        let mut energy = 0.0;
        for tex in self.textures.iter() {
            energy += tex.eval_at(lambda, uv);
        }
        energy
    }
}

impl EvalAt<f32x4> for TexStack {
    fn eval_at(&self, lambda: f32x4, uv: (f32, f32)) -> f32x4 {
        let mut energy = f32x4::ZERO;
        for tex in self.textures.iter() {
            energy += tex.eval_at(lambda, uv);
        }
        energy
    }
}
