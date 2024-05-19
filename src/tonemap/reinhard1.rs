use super::Tonemapper;
use crate::prelude::*;

use log_once::warn_once;

#[derive(Clone, Debug)]
pub struct Reinhard1 {
    key_value: f32,
    max_white: f32,
    l_w: Option<f32>,
    silenced: bool,
}

impl Reinhard1 {
    const DELTA: f32 = 0.001;
    pub fn new(key_value: f32, max_white: f32, silenced: bool) -> Self {
        Self {
            key_value,
            max_white,
            l_w: None,
            silenced,
        }
    }
}

impl Tonemapper for Reinhard1 {
    fn initialize(&mut self, film: &Vec2D<XYZColor>, factor: f32) {
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
                sum_of_log += ((Self::DELTA + lum) as f64).ln();
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

        if min_luminance < Self::DELTA {
            if !self.silenced {
                warn!("clamping min_luminance to avoid taking log(0) == NEG_INFINITY")
            };
            min_luminance = Self::DELTA;
        }

        let dynamic_range = max_luminance.log10() - min_luminance.log10();

        let avg_luminance = total_luminance / (total_pixels as f32);
        let l_w = (sum_of_log / (total_pixels as f64)).exp() as f32 / factor;

        if !self.silenced {
            info!(
                "computed tonemapping: avg luminance {}, l_w = {:?} (avg_log = {})",
                avg_luminance,
                l_w,
                sum_of_log / total_pixels as f64
            );
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
        self.l_w = Some(l_w)
    }
    fn map(&self, film: &Vec2D<XYZColor>, pixel: (usize, usize)) -> XYZColor {
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
            cie_xyz_color = MAUVE;
        }

        XYZColor::from_raw(f32x4::splat(scaling_factor) * cie_xyz_color.0)
    }

    fn get_name(&self) -> &str {
        "reinhard1"
    }
}

#[derive(Clone, Debug)]
pub struct Reinhard1x3 {
    key_value: f32x4,
    max_white: f32x4,
    l_w: Option<f32x4>,
    silenced: bool,
}

impl Reinhard1x3 {
    const DELTA: f32 = 0.001;
    pub fn new(key_value: f32, max_white: f32, silenced: bool) -> Self {
        Self {
            key_value: f32x4::splat(key_value),
            max_white: f32x4::splat(max_white),
            l_w: None,
            silenced,
        }
    }
}

impl Tonemapper for Reinhard1x3 {
    fn initialize(&mut self, film: &Vec2D<XYZColor>, factor: f32) {
        let mut max_luminance = 0.0;
        let mut min_luminance = INFINITY;
        let mut max_lum_xy = (0, 0);
        let mut min_lum_xy = (0, 0);
        let mut total_luminance = 0.0;

        let total_pixels = film.width * film.height;
        let mut sum_of_log = f32x4::splat(0.0);

        for y in 0..film.height {
            for x in 0..film.width {
                let color = film.at(x, y);
                let lum: f32 = color.y();
                debug_assert!(!lum.is_nan(), "nan {:?} at ({},{})", color, x, y);
                if lum.is_nan() {
                    continue;
                }
                total_luminance += lum;
                sum_of_log += (f32x4::splat(Self::DELTA) + color.0).ln();
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
            if !self.silenced {
                warn!("clamping min_luminance to avoid taking log(0) == NEG_INFINITY");
            }
            min_luminance = Self::DELTA as f32;
        }

        let dynamic_range = max_luminance.log10() - min_luminance.log10();

        let avg_luminance = total_luminance / (total_pixels as f32);
        let l_w = (sum_of_log / (f32x4::splat(total_pixels as f32))).exp() / f32x4::splat(factor);
        if !self.silenced {
            info!(
                "computed tonemapping: avg luminance {}, l_w = {:?} (avg_log = {:?})",
                avg_luminance,
                l_w,
                sum_of_log / f32x4::splat(total_pixels as f32)
            );
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
        self.l_w = Some(l_w)
    }
    fn map(&self, film: &Vec2D<XYZColor>, pixel: (usize, usize)) -> XYZColor {
        // TODO: determine whether applying this per channel in xyz space is adequate,
        // or if this needs to be applied in sRGB linear or cie RGB space.
        let cie_xyz_color = film.at(pixel.0, pixel.1);

        // using slightly more complex reinhard mapping
        // a = key_value
        // l = a * l / l_w
        // Ld = L(1 + L / L_max^2) / (1 + L)
        let l = self.key_value * cie_xyz_color.0
            / self
                .l_w
                .expect("tonemapper data not set correctly. was this tonemapper initialized?");
        let mul = self.max_white.powf(f32x4::splat(2.0)).recip();
        let one_l_lm2 = mul * l + f32x4::ONE;

        let scaling_factor = l * one_l_lm2 / (f32x4::ONE + l);

        // the last lane likely got set to nan or something nonzero when the division by l_w occurred, so set it to 0
        let mut mapped = (scaling_factor * cie_xyz_color.0) * Vec3::MASK;
        if !mapped.is_finite().all() || mapped.is_nan().any() {
            warn_once!("detected nan or infinity value in film after tonemapping");
            mapped = MAUVE.0;
        }
        XYZColor::from_raw(mapped)
    }

    fn get_name(&self) -> &str {
        "reinhard1x3"
    }
}
