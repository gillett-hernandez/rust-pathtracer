use crate::prelude::*;

use packed_simd::f32x4;

use super::Tonemapper;

#[derive(Clone, Debug)]
pub struct Reinhard0 {
    key_value: f32,
    l_w: Option<f32>,
    silenced: bool,
}

impl Reinhard0 {
    const DELTA: f64 = 0.001;
    pub fn new(key_value: f32, silenced: bool) -> Self {
        Self {
            key_value,
            l_w: None,
            silenced,
        }
    }
}

impl Tonemapper for Reinhard0 {
    fn initialize(&mut self, film: &Film<XYZColor>, factor: f32) {
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
            if !self.silenced {
                warn!("clamping min_luminance to avoid taking log(0) == NEG_INFINITY");
            }
            min_luminance = Self::DELTA as f32;
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
            info!("dynamic range is {}", dynamic_range);
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
    fn map(&self, film: &Film<XYZColor>, pixel: (usize, usize)) -> f32x4 {
        let mut cie_xyz_color = film.at(pixel.0, pixel.1);
        let lum = cie_xyz_color.y();

        // using super basic reinhard mapping
        // Ld = L / (1 + L)
        let l = self.key_value * lum
            / self
                .l_w
                .expect("tonemapper data not set correctly. was this tonemapper initialized?");
        let scaling_factor = l / (1.0 + l);

        if !cie_xyz_color.0.is_finite().all() || cie_xyz_color.0.is_nan().any() {
            cie_xyz_color = MAUVE;
        }

        scaling_factor * cie_xyz_color.0
    }

    fn get_name(&self) -> &str {
        "reinhard0"
    }
}

#[derive(Clone, Debug)]
pub struct Reinhard0x3 {
    key_value: f32,
    l_w: Option<f32x4>,
    silenced: bool,
}

impl Reinhard0x3 {
    const DELTA: f32 = 0.001;
    pub fn new(key_value: f32, silenced: bool) -> Self {
        Self {
            key_value,
            l_w: None,
            silenced,
        }
    }
}

impl Tonemapper for Reinhard0x3 {
    fn initialize(&mut self, film: &Film<XYZColor>, factor: f32) {
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
                let lum = color.y();

                debug_assert!(!lum.is_nan(), "nan {:?} at ({},{})", color, x, y);
                if lum.is_nan() {
                    continue;
                }
                total_luminance += lum;
                sum_of_log += (Self::DELTA + color.0).ln();
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
        let l_w = (sum_of_log / (total_pixels as f32)).exp() / factor;
        if !self.silenced {
            info!(
                "computed tonemapping: avg luminance {}, l_w = {:?} (avg_log = {:?})",
                avg_luminance,
                l_w,
                sum_of_log / total_pixels as f32
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
        }
        self.l_w = Some(l_w)
    }
    fn map(&self, film: &Film<XYZColor>, pixel: (usize, usize)) -> f32x4 {
        // TODO: determine whether applying this per channel in xyz space is adequate,
        // or if this needs to be applied in sRGB linear or cie RGB space.
        let mut cie_xyz_color = film.at(pixel.0, pixel.1);

        // using super basic reinhard mapping
        // Ld = L / (1 + L)
        let l = self.key_value * cie_xyz_color.0
            / self
                .l_w
                .expect("tonemapper data not set correctly. was this tonemapper initialized?");
        let scaling_factor = l / (1.0 + l);

        if !cie_xyz_color.0.is_finite().all() || cie_xyz_color.0.is_nan().any() {
            cie_xyz_color = MAUVE;
        }

        scaling_factor * cie_xyz_color.0
    }

    fn get_name(&self) -> &str {
        "reinhard0x3"
    }
}
