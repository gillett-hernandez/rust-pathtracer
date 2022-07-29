#![allow(unused, unused_imports)]
use crate::curves::mauve;
use crate::renderer::Film;
use math::{SpectralPowerDistributionFunction, XYZColor, INFINITY};

use nalgebra::{Matrix3, Vector3};
use packed_simd::f32x4;

use std::time::Instant;

use super::Tonemapper;

#[derive(Clone, Debug)]
pub struct Reinhard1 {
    key_value: f32,
    max_white: f32,
    l_w: Option<f32>,
}

impl Reinhard1 {
    const DELTA: f64 = 0.001;
    pub fn new(key_value: f32, max_white: f32) -> Self {
        Self {
            key_value,
            max_white,
            l_w: None,
        }
    }
}

impl Tonemapper for Reinhard1 {
    fn initialize(&mut self, film: &Film<XYZColor>) {
        let mut max_luminance = 0.0;
        let mut min_luminance = INFINITY;
        let mut max_lum_xy = (0, 0);
        let mut min_lum_xy = (0, 0);
        let mut total_luminance = 0.0;

        let total_pixels = film.width * film.height;
        let mut sum_of_log = 0.0f64;

        for y in 0..film.height {
            for x in 0..film.width {
                let color = film.at(x, y);
                let lum: f32 = color.y();
                debug_assert!(!lum.is_nan(), "nan {:?} at ({},{})", color, x, y);
                if lum.is_nan() {
                    continue;
                }
                total_luminance += lum;
                sum_of_log += (Self::DELTA + lum as f64).ln();
                if lum > max_luminance {
                    // println!("max lum {} at ({}, {})", max_luminance, x, y);
                    max_luminance = lum;
                    max_lum_xy = (x, y);
                }
                if lum < min_luminance {
                    min_luminance = lum;
                    min_lum_xy = (x, y);
                }
            }
        }

        if min_luminance < Self::DELTA as f32 {
            warn!("clamping min_luminance to avoid taking log(0) == NEG_INFINITY");
            min_luminance = Self::DELTA as f32;
        }

        let dynamic_range = max_luminance.log10() - min_luminance.log10();

        let avg_luminance = total_luminance / (total_pixels as f32);
        let l_w = (sum_of_log / (total_pixels as f64)).exp() as f32;

        info!(
            "computed tonemapping: avg luminance {}, l_w = {:?} (avg_log = {})",
            avg_luminance,
            l_w,
            sum_of_log / total_pixels as f64
        );
        info!("dynamic range is {}", dynamic_range);
        info!(
            "max luminance occurred at {}, {}, is {}",
            max_lum_xy.0, max_lum_xy.1, max_luminance
        );
        info!(
            "min luminance occurred at {}, {}, is {}",
            min_lum_xy.0, min_lum_xy.1, min_luminance
        );
        self.l_w = Some(l_w)
    }
    fn map(&self, film: &Film<XYZColor>, pixel: (usize, usize)) -> f32x4 {
        let mut cie_xyz_color = film.at(pixel.0, pixel.1);
        let lum = cie_xyz_color.y();

        // using slightly more complex reinhard mapping
        // a = key_value
        // l = a * l / l_w
        // Ld = L(1 + L / L_max^2) / (1 + L)
        let l = self.key_value * lum
            / self
                .l_w
                .expect("tonemapper data not set correctly. was this tonemapper initialized?");
        let mul = self.max_white.powi(2).recip();
        let one_l_lm2 = mul * l + 1.0;

        let scaling_factor = l * one_l_lm2 / (1.0 + l);

        if !cie_xyz_color.0.is_finite().all() || cie_xyz_color.0.is_nan().any() {
            // mauve. universal sign of danger
            cie_xyz_color =
                XYZColor::new(0.51994668667475025, 51.48686803771597, 1.0180528737469167);
        }

        scaling_factor * cie_xyz_color.0
    }
}
