#![allow(unused, unused_imports)]
use crate::config::RenderSettings;
use crate::math::XYZColor;
use crate::renderer::Film;
pub trait Tonemapper {
    fn map(film: &Film<XYZColor>, input: XYZColor) -> XYZColor;
}

pub struct GammaTonemapper {
    pub exposure: f32,
    pub gamma: f32,
}

impl GammaTonemapper {
    pub const fn new(exposure: f32, gamma: f32) -> Self {
        GammaTonemapper { exposure, gamma }
    }
    // pub fn new_from_config(settings: &RenderSettings) -> Self {}
}
