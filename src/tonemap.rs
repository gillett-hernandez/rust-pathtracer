#![allow(unused, unused_imports)]
use crate::renderer::Film;
use crate::{config::RenderSettings, curves::mauve};
use math::{SpectralPowerDistributionFunction, XYZColor, INFINITY};

extern crate exr;
use nalgebra::{Matrix3, Vector3};
use packed_simd::f32x4;

use std::time::Instant;

pub trait Tonemapper {
    fn map(&self, film: &Film<XYZColor>, pixel: (usize, usize)) -> (f32x4, f32x4);
    fn write_to_files(&self, film: &Film<XYZColor>, exr_filename: &str, png_filename: &str);
}

#[allow(non_camel_case_types)]
pub struct sRGB {
    pub l_w: f32,
}

// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB

// FIXME: refactor to implement some actual tonemap operators :facepalm:
// this currently is just a linear operator where everything is scaled down based on the max luminance.

// reference https://64.github.io/tonemapping/
// and Reinhard '01 https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf

impl sRGB {
    const DELTA: f64 = 0.001;
    pub fn new(film: &Film<XYZColor>, exposure_adjustment: f32, printout: bool) -> Self {
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
        if printout {
            println!(
                "computed tonemapping: avg luminance {}, l_w = {:?} (avg_log = {})",
                avg_luminance,
                l_w,
                sum_of_log / total_pixels as f64
            );
            println!("dynamic range is {}", dynamic_range);
            info!(
                "max luminance occurred at {}, {}, is {}",
                max_lum_xy.0, max_lum_xy.1, max_luminance
            );
            info!(
                "min luminance occurred at {}, {}, is {}",
                min_lum_xy.0, min_lum_xy.1, min_luminance
            );
        }

        // fold in the key value into l_w
        sRGB {
            l_w: l_w / exposure_adjustment,
        }
    }
}

impl Tonemapper for sRGB {
    fn map(&self, film: &Film<XYZColor>, pixel: (usize, usize)) -> (f32x4, f32x4) {
        let mut cie_xyz_color = film.at(pixel.0, pixel.1);
        let lum = cie_xyz_color.y();
        let scaling_factor = if true {
            // using super basic reinhard mapping
            // Ld = L / (1 + L)
            let l = lum / self.l_w;
            let ld = l / (1.0 + l);
            ld
        } else {
            // straight linear scaling "tonemapping"
            1.0
        };

        if !cie_xyz_color.0.is_finite().all() || cie_xyz_color.0.is_nan().any() {
            // mauve. universal sign of danger
            cie_xyz_color =
                XYZColor::new(0.51994668667475025, 51.48686803771597, 1.0180528737469167);
        }

        let xyz_to_rgb: Matrix3<f32> = Matrix3::new(
            3.24096994,
            -1.53738318,
            -0.49861076,
            -0.96924364,
            1.8759675,
            0.04155506,
            0.05563008,
            -0.20397696,
            1.05697151,
        );
        let [x, y, z, _]: [f32; 4] = cie_xyz_color.0.into();
        let intermediate = xyz_to_rgb * Vector3::new(x, y, z);

        let unmapped = f32x4::new(intermediate[0], intermediate[1], intermediate[2], 0.0);
        let to_map = scaling_factor * unmapped.clone();
        const S313: f32x4 = f32x4::splat(0.0031308);
        const S323_25: f32x4 = f32x4::splat(323.0 / 25.0);
        const S5_12: f32x4 = f32x4::splat(5.0 / 12.0);
        const S211: f32x4 = f32x4::splat(211.0);
        const S11: f32x4 = f32x4::splat(11.0);
        const S200: f32x4 = f32x4::splat(200.0);
        let srgb =
            (to_map.lt(S313)).select(S323_25 * to_map, (S211 * to_map.powf(S5_12) - S11) / S200);

        (srgb, unmapped)
    }
    fn write_to_files(&self, film: &Film<XYZColor>, exr_filename: &str, png_filename: &str) {
        let now = Instant::now();

        print!("saving exr image...");
        exr::prelude::write_rgb_file(exr_filename, film.width, film.height, |x, y| {
            let (_mapped, linear) = self.map(&film, (x, y));
            let [r, g, b, _]: [f32; 4] = linear.into();
            (r, g, b)
        })
        .unwrap();
        println!(" done!");
        let mut img: image::RgbImage =
            image::ImageBuffer::new(film.width as u32, film.height as u32);

        for (x, y, pixel) in img.enumerate_pixels_mut() {
            //apply tonemap here

            let (mapped, _linear) = self.map(&film, (x as usize, y as usize));

            let [r, g, b, _]: [f32; 4] = mapped.into();

            *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
        }
        print!("saving image...");
        img.save(png_filename).unwrap();
        println!(" done!");

        println!(
            "took {}s to tonemap and output\n",
            (now.elapsed().as_millis() as f32) / 1000.0
        );
    }
}
