#![feature(portable_simd)]
use std::path::Path;

use rust_pathtracer as root;

use exr::prelude::*;
use rayon::prelude::*;
use root::{
    prelude::{f32x4, SimdFloat},
    vec2d::Vec2D,
};
use clap::Parser;

fn read_exr_file<P: AsRef<Path>>(name: P) -> Option<Vec2D<f32x4>> {
    read_first_rgba_layer_from_file(
        name,
        |size, _| Vec2D::new(size.0, size.1, f32x4::splat(0.0)),
        |film, pos, (r, g, b, a)| {
            film.write_at(pos.0, pos.1, f32x4::from_array([r, g, b, a]));
        },
    )
    .ok()
    .map(|e| e.layer_data.channel_data.pixels)
}

#[derive(Debug, Parser)]
struct Opt {
    #[arg(long)]
    pub compare_file: String,
    #[arg(long)]
    pub ground_truth_file: String,
    #[arg(long)]
    pub output_file: String,
    #[arg(long, default_value = "absolute_difference")]
    pub mode: String,
}

#[derive(Copy, Clone)]
enum Mode {
    AbsoluteDifference,
    RMSE,
    RelativeError,
}

impl Mode {
    pub fn new(mode: &str) -> Mode {
        match mode {
            "rmse" => Mode::RMSE,
            "relative" => Mode::RelativeError,
            _ => Mode::AbsoluteDifference,
        }
    }
}

fn main() {
    let opts = Opt::parse();

    let maybe_image0 = read_exr_file(opts.compare_file);
    let maybe_image1 = read_exr_file(opts.ground_truth_file);

    let mode = Mode::new(&opts.mode);

    if let (Some(mut image0), Some(image1)) = (maybe_image0, maybe_image1) {
        let (width, height) = (image0.width, image0.height);
        assert!(
            width == image1.width && height == image1.height,
            "image dimensions must match"
        );
        // images are in linear sRGB space
        // however since we're just trying to determine the RMSE of image0 compared to image1, it doesn't really matter that much
        match mode {
            Mode::AbsoluteDifference => {
                image0
                    .buffer
                    .par_iter_mut()
                    .zip(image1.buffer.par_iter())
                    .for_each(|(px0, px1)| {
                        // absolute difference per channel
                        *px0 = (*px0 - *px1).abs();
                    });

                write_rgb_file(&opts.output_file, image0.width, image0.height, |x, y| {
                    // TODO: maybe don't use linear srgb here, but instead CIE RGB
                    let [r, g, b, _]: [f32; 4] = image0.at(x, y).into();
                    (r, g, b)
                })
                .unwrap();
            }
            Mode::RMSE => {
                let g = colorgrad::viridis();
                // pass 1: calculate rmse
                image0
                    .buffer
                    .par_iter_mut()
                    .zip(image1.buffer.par_iter())
                    .for_each(|(px0, px1)| {
                        // rmse. not a true rmse since a true rmse would take into account the error of all the constituent samples that go into a pixel
                        let difference = *px0 - *px1;
                        let sum_of_sqr = (difference * difference).reduce_sum() / 4.0;
                        let rmse = sum_of_sqr.sqrt();
                        *px0 = f32x4::splat(rmse);
                        px0[3] = 0.0;
                    });
                // grab max rmse

                // "tonemap" rmse
                let max_rmse = image0.buffer.iter().map(|e| e[0]).reduce(f32::max).unwrap();
                let min_rmse = image0.buffer.iter().map(|e| e[0]).reduce(f32::min).unwrap();
                println!("minmax: {} -> {}", min_rmse, max_rmse);

                image0
                    .buffer
                    .par_iter_mut()
                    // .zip(image1.buffer.par_iter())
                    .for_each(|px| {
                        // rmse. not a true rmse since a true rmse would take into account the error of all the constituent samples that go into a pixel
                        let rmse = px[0];

                        let color = g.at(((rmse - min_rmse) / (max_rmse - min_rmse)) as f64);

                        *px = f32x4::from_array([
                            color.r as f32,
                            color.g as f32,
                            color.b as f32,
                            color.a as f32,
                        ]);
                    });

                let mut img: image::RgbImage = image::ImageBuffer::new(width as u32, height as u32);

                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    //apply tonemap here

                    let [r, g, b, _]: [f32; 4] = image0.at(x as usize, y as usize).into();

                    *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
                }
                let mut png_filename = opts.output_file;
                if png_filename.ends_with(".exr") {
                    png_filename.truncate(png_filename.len() - 4);
                }
                png_filename.push_str(".png");

                img.save(png_filename).unwrap();
            }
            Mode::RelativeError => {
                image0
                    .buffer
                    .par_iter_mut()
                    .zip(image1.buffer.par_iter())
                    .for_each(|(px0, px1)| {
                        // absolute error per channel
                        let absolute_error = (*px0 - *px1).abs();
                        // relative error per channel
                        let relative_error = absolute_error / *px1;
                        *px0 = relative_error
                            .is_finite()
                            .select(relative_error, f32x4::splat(0.0));
                    });

                write_rgb_file(&opts.output_file, image0.width, image0.height, |x, y| {
                    // TODO: maybe don't use linear srgb here, but instead CIE RGB
                    let [r, g, b, _]: [f32; 4] = image0.at(x, y).into();
                    (r, g, b)
                })
                .unwrap();
            }
        }

        println!("saved, exiting");
    } else {
        println!("failed to parse images for some reason. check whether the paths exist");
    }
}
