use crate::math::*;
use crate::renderer::Film;

// use std::error::Error;
// use std::fs::File;
// use std::io::Read;

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
}

#[derive(Clone)]
pub struct TexStack {
    pub textures: Vec<Texture>,
}
impl TexStack {
    pub fn eval_at(&self, lambda: f32, uv: (f32, f32)) -> f32 {
        let mut s = 0.0;
        for tex in self.textures.iter() {
            s += tex.eval_at(lambda, uv);
        }
        s
    }
}
