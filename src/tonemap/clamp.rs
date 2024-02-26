use super::Tonemapper;
use crate::prelude::*;

use packed_simd::f32x4;

#[derive(Clone, Debug)]
pub struct Clamp {
    exposure: f32,
    luminance_only: bool,
    factor: f32,
    silenced: bool,
}

impl Clamp {
    pub fn new(exposure: f32, luminance_only: bool, silenced: bool) -> Self {
        Self {
            exposure,
            luminance_only,
            factor: 1.0,
            silenced,
        }
    }
}

impl Tonemapper for Clamp {
    fn initialize(&mut self, film: &Vec2D<XYZColor>, factor: f32) {
        self.factor = factor;
        let mut max_luminance = 0.0;
        let mut min_luminance = INFINITY;
        let mut max_lum_xy = (0, 0);
        let mut min_lum_xy = (0, 0);
        let mut total_luminance = 0.0;

        let total_pixels = film.width * film.height;

        for y in 0..film.height {
            for x in 0..film.width {
                let color = film.at(x, y);
                let lum: f32 = color.y();
                debug_assert!(!lum.is_nan(), "nan {:?} at ({},{})", color, x, y);
                if lum.is_nan() {
                    continue;
                }
                total_luminance += lum;
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

        if min_luminance < 0.00001 {
            if !self.silenced {
                warn!("clamping min_luminance to avoid taking log(0) == NEG_INFINITY");
            }
            min_luminance = 0.00001;
        }

        let dynamic_range = max_luminance.log10() - min_luminance.log10();

        let _avg_luminance = total_luminance / (total_pixels as f32);
        if !self.silenced {
            info!("dynamic range is {} dB", dynamic_range);
            info!(
                "max luminance occurred at {}, {}, is {}",
                max_lum_xy.0, max_lum_xy.1, max_luminance
            );
            info!(
                "min luminance occurred at {}, {}, is {}",
                min_lum_xy.0, min_lum_xy.1, min_luminance
            );
        }
    }
    fn map(&self, film: &Vec2D<XYZColor>, pixel: (usize, usize)) -> XYZColor {
        let mut cie_xyz_color = film.at(pixel.0, pixel.1) * self.factor;
        if !cie_xyz_color.0.is_finite().all() || cie_xyz_color.0.is_nan().any() {
            cie_xyz_color = MAUVE;
        }

        let exposure_mult = 2.0f32.powf(self.exposure);
        if self.luminance_only {
            let lum = cie_xyz_color.y();

            // need to scale and clamp the color
            let new_lum = (lum * exposure_mult).clamp(0.0, 1.0);
            // actual factor
            let scaling_factor = new_lum / lum;
            // TODO: determine if this needs to be done per channel.

            XYZColor::from_raw(scaling_factor * cie_xyz_color.0)
        } else {
            XYZColor::from_raw(
                (cie_xyz_color * exposure_mult)
                    .0
                    .min(f32x4::splat(1.0))
                    .max(f32x4::splat(0.0)),
            )
        }
    }
    fn get_name(&self) -> &str {
        "clamp"
    }
}
