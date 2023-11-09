use crate::prelude::*;

use nalgebra::{Matrix3, Vector3};
use packed_simd::f32x4;

use std::{error::Error, time::Instant};

mod clamp;
mod reinhard0;
mod reinhard1;

pub use clamp::Clamp;
pub use reinhard0::{Reinhard0, Reinhard0x3};
pub use reinhard1::{Reinhard1, Reinhard1x3};

// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
const XYZ_TO_SRGB_LINEAR: Matrix3<f32> = Matrix3::new(
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
const S313: f32x4 = f32x4::splat(0.0031308);
const S323_25: f32x4 = f32x4::splat(323.0 / 25.0);
const S5_12: f32x4 = f32x4::splat(5.0 / 12.0);
const S211: f32x4 = f32x4::splat(211.0);
const S11: f32x4 = f32x4::splat(11.0);
const S200: f32x4 = f32x4::splat(200.0);

// reference https://64.github.io/tonemapping/
// and Reinhard '02 https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf

pub trait Tonemapper: Send + Sync {
    // factor in `initialize` is a prefactor weight on the film contents.
    // a tonemapper should initialize based on the film x scale_factor
    // and should premultiply the film values when tonemapping through `map`
    fn initialize(&mut self, film: &Vec2D<XYZColor>, factor: f32);
    // should tonemap a pixel from hdr to ldr
    fn map(&self, film: &Vec2D<XYZColor>, pixel: (usize, usize)) -> f32x4;
    fn get_name(&self) -> &str;
}

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub enum Converter {
    sRGB,
}

impl Converter {
    pub fn transfer_function(&self, color: f32x4, linear: bool) -> f32x4 {
        match self {
            Converter::sRGB => {
                let [x, y, z, _]: [f32; 4] = color.into();
                let intermediate = XYZ_TO_SRGB_LINEAR * Vector3::new(x, y, z);

                let srgb_linear =
                    f32x4::new(intermediate[0], intermediate[1], intermediate[2], 0.0);
                if linear {
                    srgb_linear
                } else {
                    (srgb_linear.lt(S313)).select(
                        S323_25 * srgb_linear,
                        (S211 * srgb_linear.powf(S5_12) - S11) / S200,
                    )
                }
            }
        }
    }

    pub fn write_to_files(
        &self,
        film: &Vec2D<XYZColor>,
        tonemapper: &Box<dyn Tonemapper>,
        factor: f32,
        exr_filename: &str,
        png_filename: &str,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Converter::sRGB => {
                let now = Instant::now();

                print!("saving exr...");
                exr::prelude::write_rgb_file(exr_filename, film.width, film.height, |x, y| {
                    // TODO: maybe don't use linear srgb here, but instead CIE RGB
                    let [r, g, b, _]: [f32; 4] = self
                        .transfer_function(film.at(x, y).0 * factor, true)
                        .into();
                    (r, g, b)
                })?;
                println!(" done!");
                let mut img: image::RgbImage =
                    image::ImageBuffer::new(film.width as u32, film.height as u32);

                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    //apply tonemap here

                    let [r, g, b, _]: [f32; 4] = self
                        .transfer_function(tonemapper.map(film, (x as usize, y as usize)), false)
                        .into();

                    *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
                }
                print!("saving image...");
                img.save(png_filename)?;
                println!(" done!");

                println!(
                    "took {}s to tonemap and output\n",
                    (now.elapsed().as_millis() as f32) / 1000.0
                );
                Ok(())
            }
        }
    }
}

// should the tonemapper own a converter, or should a converter own the townmapper

#[cfg(test)]
mod test {

    use rand::random;

    use crate::tonemap::{Clamp, Reinhard0, Reinhard0x3, Reinhard1, Reinhard1x3};

    use super::*;
    #[test]
    fn test_write_to_file() {
        let num_samples = random::<f32>() * 1000.0 + 1.0;

        let mut tonemappers: Vec<Box<dyn Tonemapper>> = vec![
            Box::new(Clamp::new(0.0, true, false)),
            Box::new(Clamp::new(0.0, false, false)),
            Box::new(Reinhard0::new(0.18, false)),
            Box::new(Reinhard0x3::new(0.18, false)),
            Box::new(Reinhard1::new(0.18, 10.0, false)),
            Box::new(Reinhard1x3::new(0.18, 10.0, false)),
        ];

        let mut film = Vec2D::new(1024, 1024, XYZColor::BLACK);

        // estimated total energy per pixel is proportional to num_samples
        println!("adding {} samples per pixel", num_samples as usize);
        film.buffer.par_iter_mut().for_each(|px| {
            for _ in 0..(num_samples as usize) {
                *px += SingleWavelength::new_from_range(random::<f32>(), BOUNDED_VISIBLE_RANGE)
                    .replace_energy(1.0)
                    .into();
            }
        });

        let converter = Converter::sRGB;
        for (i, tonemapper) in tonemappers.iter_mut().enumerate() {
            let name = tonemapper.get_name();
            println!("tonemapper is {}", name);

            let exr_filename = format!("test_output_{}_{}_{}.exr", num_samples as usize, name, i);
            let png_filename = format!("test_output_{}_{}_{}.png", num_samples as usize, name, i);
            tonemapper.initialize(&film, 1.0 / num_samples);
            converter
                .write_to_files(
                    &film,
                    tonemapper,
                    1.0 / num_samples,
                    &exr_filename,
                    &png_filename,
                )
                .expect("failed to write files");
        }
    }
}
