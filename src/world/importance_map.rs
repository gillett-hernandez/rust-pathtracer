use std::sync::Arc;

use math::curves::{InterpolationMode, Op};

use parking_lot::Mutex;

use crate::texture::TexStack;
use crate::{math::*, rgb_to_u32};

use minifb::{Scale, Window, WindowOptions};
use pbr::ProgressBar;

use rayon::iter::ParallelIterator;
use rayon::prelude::*;

// equirectangular importance map.
// maybe refactor this to another submodule under or separate from src/world so that more importance map types can be defined.
// also, refactor this importance map type so that it can properly sample from the curves that make up texstack
// would change memory usage complexity from O(n*m) to O(k*n*m) where k is the number of channels in the texstack
// which is 4 for every tex4, 1 for every tex1, etc.

#[derive(Clone)]
pub enum ImportanceMap {
    Baked {
        luminance_curve: Curve,
        vertical_resolution: usize,
        horizontal_resolution: usize,
        data: Vec<CurveWithCDF>,
        marginal_cdf: CurveWithCDF,
    },
    Unbaked {
        luminance_curve: Curve,
        vertical_resolution: usize,
        horizontal_resolution: usize,
    },
    Empty,
}

impl ImportanceMap {
    pub fn bake_in_place(&mut self, texture: &TexStack, wavelength_bounds: Bounds1D) -> bool {
        warn!(
            "baking importance map with wavelength_bounds {:?}",
            wavelength_bounds
        );
        match self {
            Self::Baked {
                luminance_curve,
                vertical_resolution,
                horizontal_resolution,
                ..
            }
            | Self::Unbaked {
                luminance_curve,
                vertical_resolution,
                horizontal_resolution,
            } => {
                *self = Self::bake_raw(
                    texture,
                    *vertical_resolution,
                    *horizontal_resolution,
                    luminance_curve.clone(),
                    wavelength_bounds,
                );
                true
            }
            _ => false,
        }
    }
    pub fn bake_raw(
        texture_stack: &TexStack,
        vertical_resolution: usize,
        horizontal_resolution: usize,
        luminance_curve: Curve,
        wavelength_bounds: Bounds1D,
    ) -> Self {
        let num_samples_for_texel_spectra = 100;
        let mut data = Vec::new();
        let mut marginal_data = Vec::new();

        let mut total_luminance = 0.0;

        // let mut machine = SPD::Machine {
        //     seed: 1.0,
        //     list: vec![(Op::Mul, luminance_curve), (Op::Mul, SPD::Const(0.0))],
        // };

        let (mut window, mut buffer, mut maybe_cdf) = if cfg!(feature = "visualize_importance_map")
        {
            println!("visualize feature enabled");
            let mut window = Window::new(
                "Preview",
                horizontal_resolution,
                vertical_resolution,
                WindowOptions {
                    scale: Scale::X1,
                    ..WindowOptions::default()
                },
            )
            .unwrap_or_else(|e| {
                panic!("{}", e);
            });

            window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));
            (
                Some(window),
                vec![0u32; horizontal_resolution * vertical_resolution],
                Some(Vec::new()),
            )
        } else {
            (None, vec![], None)
        };

        println!("generating importance map");
        let pb = Arc::new(Mutex::new(ProgressBar::new(
            (vertical_resolution * horizontal_resolution) as u64,
        )));

        // construct CDF from each row
        let mut rows: Vec<(CurveWithCDF, f32)> = (0..vertical_resolution)
            .into_par_iter()
            .map(|row| -> (CurveWithCDF, f32) {
                let mut signal = Vec::new();
                // cumulative mass function, discrete equivalent of the CDF.
                let mut signal_cmf = Vec::new();
                let mut row_luminance = 0.0;
                for column in 0..horizontal_resolution {
                    let uv = (
                        row as f32 / vertical_resolution as f32,
                        column as f32 / horizontal_resolution as f32,
                    );
                    let machine = Curve::Machine {
                        seed: 1.0,
                        list: vec![
                            (Op::Mul, luminance_curve.clone()),
                            (Op::Mul, texture_stack.curve_at(uv)),
                        ],
                    };
                    let texel_luminance = machine.evaluate_integral(
                        wavelength_bounds,
                        num_samples_for_texel_spectra,
                        false,
                    );
                    row_luminance += texel_luminance;
                    signal.push(texel_luminance);
                    signal_cmf.push(row_luminance);
                }
                // normalize pdf (and cdf?)
                signal.iter_mut().for_each(|e| {
                    *e /= row_luminance;
                });
                signal_cmf.iter_mut().for_each(|e| {
                    *e /= row_luminance;
                });
                pb.lock().add(horizontal_resolution as u64);
                (
                    CurveWithCDF {
                        pdf: Curve::Linear {
                            signal,
                            bounds: Bounds1D::new(0.0, 1.0),
                            mode: InterpolationMode::Nearest,
                        },
                        cdf: Curve::Linear {
                            signal: signal_cmf,
                            bounds: Bounds1D::new(0.0, 1.0),
                            mode: InterpolationMode::Nearest,
                        },
                        pdf_integral: 1.0,
                    },
                    row_luminance,
                )
            })
            .collect();
        for (row, (cdf, row_luminance)) in rows.drain(..).enumerate() {
            // let  = rows[row];
            if cfg!(feature = "visualize_importance_map") {
                for column in 0..horizontal_resolution {
                    let rgb = (cdf
                        .cdf
                        .evaluate_power(column as f32 / horizontal_resolution as f32))
                    .clamp(0.0, 1.0 - std::f32::EPSILON);
                    buffer[row * horizontal_resolution + column] = rgb_to_u32(
                        (rgb * 256.0) as u8,
                        (rgb * 256.0) as u8,
                        (rgb * 256.0) as u8,
                    );
                }
            }
            total_luminance += row_luminance;
            data.push(cdf);
            marginal_data.push(row_luminance);
            maybe_cdf.iter_mut().for_each(|e| e.push(total_luminance));

            if let Some(window) = &mut window {
                if window.is_open() {
                    window
                        .update_with_buffer(&buffer, horizontal_resolution, vertical_resolution)
                        .unwrap();
                }
            }
        }
        pb.lock().finish();
        println!("");

        marginal_data.iter_mut().for_each(|e| *e /= total_luminance);

        if let Some(window) = &mut window {
            let v_cdf = maybe_cdf.unwrap();
            for row in 0..vertical_resolution {
                if !window.is_open() {
                    break;
                }
                let rgb = (v_cdf[row] / total_luminance).clamp(0.0, 1.0 - std::f32::EPSILON);
                let u32 = rgb_to_u32(
                    (rgb * 256.0) as u8,
                    (rgb * 256.0) as u8,
                    (rgb * 256.0) as u8,
                );
                buffer[row * horizontal_resolution] = u32;
                buffer[row * horizontal_resolution + 1] = u32;
                buffer[row * horizontal_resolution + 2] = u32;
                window
                    .update_with_buffer(&buffer, horizontal_resolution, vertical_resolution)
                    .unwrap();
            }
        }
        let marginal_cdf = Curve::Linear {
            signal: marginal_data,
            bounds: Bounds1D::new(0.0, 1.0),
            mode: InterpolationMode::Nearest,
        }
        .to_cdf(Bounds1D::new(0.0, 1.0), 100);

        Self::Baked {
            data,
            marginal_cdf,
            horizontal_resolution,
            vertical_resolution,
            luminance_curve,
        }
    }
    pub fn sample_uv(&self, sample: Sample2D) -> ((f32, f32), (PDF, PDF)) {
        match self {
            Self::Baked {
                data, marginal_cdf, ..
            } => {
                // inverse transform sample of the marginal distribution, selecting a row with a high luminance.
                let (
                    SingleWavelength {
                        lambda: u,
                        energy: _,
                    },
                    row_pdf,
                ) = marginal_cdf
                    .sample_power_and_pdf(Bounds1D::new(0.0, 1.0), Sample1D::new(sample.y));

                // inverse transform sample the selected row, finding a uv coordinate with a high luminance.
                let (
                    SingleWavelength {
                        lambda: v,
                        energy: _,
                    },
                    column_pdf,
                ) = data[(u * data.len() as f32) as usize]
                    .sample_power_and_pdf(Bounds1D::new(0.0, 1.0), Sample1D::new(sample.x));
                assert!(u < 1.0, "{}", u);
                assert!(u >= 0.0, "{}", u);
                assert!(v < 1.0, "{}", v);
                assert!(v >= 0.0, "{}", v);
                ((u, v), (row_pdf, column_pdf))
            }
            _ => panic!("used unbaked importance map"),
        }
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use math::spectral::{y_bar, BOUNDED_VISIBLE_RANGE};

    use super::*;
    use crate::renderer::Film;
    use crate::tonemap::{Clamp, Converter, Tonemapper};

    use crate::world::environment::*;
    use crate::{
        parsing::{construct_world, parse_tonemapper},
        texture::{Texture, Texture1},
    };

    #[test]
    fn test_raw_importance_map() {
        let mut film = Film::new(512, 512, 0.0f32);

        for (idx, pixel) in film.buffer.iter_mut().enumerate() {
            let (x, y) = (idx % film.width, (idx / film.width));
            if (x as f32 - 200.0).powi(2) + (y as f32 - 200.0).powi(2) < 400.0 {
                *pixel += 1000.0;
            }
            *pixel += random();
        }

        let texture = TexStack {
            textures: vec![Texture::Texture1(Texture1 {
                curve: Curve::y_bar().to_cdf(BOUNDED_VISIBLE_RANGE, 100),
                texture: film,
                interpolation_mode: InterpolationMode::Nearest,
            })],
        };
        let map =
            ImportanceMap::bake_raw(&texture, 1024, 1024, Curve::y_bar(), BOUNDED_VISIBLE_RANGE);

        // let mut sum = 0.0;
        let width = 512;
        let height = 256;
        let limit = width * height;
        let deterministic = false;
        // let limit = 100000;
        let mut estimate = 0.0;
        let mut pb = ProgressBar::new(limit as u64);

        let mut window = Window::new(
            "Preview",
            width,
            height,
            WindowOptions {
                scale: Scale::X2,
                ..WindowOptions::default()
            },
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });
        let mut buffer = vec![0u32; limit];
        let mut film = Film::new(width, height, XYZColor::BLACK);

        let converter = Converter::sRGB;

        let mut tonemapper = Clamp::new(0.0);

        window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

        for idx in 0..limit {
            if !window.is_open() {
                println!("window closed, stopping test");
                break;
            }

            let sample = Sample2D::new_random_sample();
            let x_float = ((idx % width) as f32 + sample.x) / width as f32;
            let y_float = ((idx / width) as f32 + sample.y) / height as f32;

            let sample = if deterministic {
                Sample2D::new(x_float, y_float)
            } else {
                sample
            };
            let (uv, pdf) = map.sample_uv(sample);
            let pdf = (pdf.0 * pdf.1).0;

            // estimate of env map luminance will have unacceptable bias depending on the actual size of the env map texture.
            // need to downsample to retain size information in importance map so that the pdf can be adjusted.

            for _ in 0..4 {
                let (mut sw, _) = (
                    SingleWavelength::new_from_range(0.5, BOUNDED_VISIBLE_RANGE),
                    1.0,
                );

                sw.energy.0 = texture.eval_at(sw.lambda, uv);

                // sum += y_bar(sw.lambda * 10.0) * sw.energy.0;
                estimate += y_bar(sw.lambda * 10.0) * sw.energy.0 / pdf;
                let (px, py) = (
                    (uv.0 * width as f32) as usize,
                    (uv.1 * height as f32) as usize,
                );

                // film.buffer[px + width * py] += XYZColor::from(sw) / (pdf.0 + 0.01) / wavelength_pdf;
                film.buffer[px + width * py] += XYZColor::new(1.0, 1.0, 1.0) * sw.energy.0 / pdf;
            }

            if idx % 100 == 0 {
                pb.add(100);

                buffer
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(pixel_idx, v)| {
                        let y: usize = pixel_idx / width;
                        let x: usize = pixel_idx - width * y;

                        let [r, g, b, _]: [f32; 4] = converter
                            .transfer_function(
                                tonemapper.map(&film, (x as usize, y as usize)),
                                false,
                            )
                            .into();
                        *v = rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
                    });
                window.update_with_buffer(&buffer, width, height).unwrap();
            }
        }
        println!("\n\nestimate is {}", estimate / limit as f32);
    }

    #[test]
    fn test_env_importance_sampling() {
        let mut world = construct_world(PathBuf::from("data/scenes/hdri_test_2.toml")).unwrap();

        if let EnvironmentMap::HDR {
            importance_map,
            texture,
            ..
        } = &mut world.environment
        {
            importance_map.bake_in_place(texture, BOUNDED_VISIBLE_RANGE);
        }

        let env = &world.environment;
        if let EnvironmentMap::HDR {
            importance_map,
            texture,
            strength,
            ..
        } = env
        {
            if matches!(importance_map, ImportanceMap::Baked { .. }) {
                println!("got importance map, now do some math on it to see if it works");
            } else {
                println!("testing env map without importance map, now do some math on it to see if it works");
            }

            let wavelength_range = BOUNDED_VISIBLE_RANGE;

            // let mut sum = 0.0;
            let width = 512;
            let height = 256;
            let limit = width * height;
            let deterministic = false;
            // let limit = 100000;
            let mut estimate = 0.0;
            let mut estimate2 = 0.0;
            let mut pb = ProgressBar::new(limit as u64);

            let mut window = Window::new(
                "Preview",
                width,
                height,
                WindowOptions {
                    scale: Scale::X2,
                    ..WindowOptions::default()
                },
            )
            .unwrap_or_else(|e| {
                panic!("{}", e);
            });
            let mut buffer = vec![0u32; limit];
            let mut film = Film::new(width, height, XYZColor::BLACK);
            let converter = Converter::sRGB;

            let mut tonemapper = Clamp::new(0.0);

            window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

            // let mut sampler = StratifiedSampler::new(100, 100, 100);
            for idx in 0..limit {
                if !window.is_open() {
                    println!("window closed, stopping test");
                    break;
                }
                // let sample = sampler.draw_2d();
                let sample = Sample2D::new_random_sample();
                let x_float = ((idx % width) as f32 + sample.x) / width as f32;
                let y_float = ((idx / width) as f32 + sample.y) / height as f32;
                // env.sample_emission(
                //     world.get_world_radius(),
                //     world.get_center(),
                //     Sample2D::new_random_sample(),
                //     Sample2D::new_random_sample(),
                //     BOUNDED_VISIBLE_RANGE,
                //     Sample1D::new_random_sample(),
                // );

                let sample = if deterministic {
                    Sample2D::new(x_float, y_float)
                } else {
                    sample
                };
                // println!("{} {}", sample.x, sample.y);
                let (uv, pdf) = env.sample_env_uv(sample);
                let pdf_for_result = env.pdf_for(uv).0 + 0.01;

                for _ in 0..4 {
                    let wavelength_sample = if false {
                        // pick a constant wavelength to reduce variance
                        Sample1D::new(0.35)
                    } else {
                        Sample1D::new_random_sample()
                    };
                    let (mut sw, wavelength_pdf) = (
                        SingleWavelength::new_from_range(wavelength_sample.x, wavelength_range),
                        1.0 / wavelength_range.span(),
                    );

                    sw.energy.0 = texture.eval_at(sw.lambda, uv) * strength;

                    // sum += y_bar(sw.lambda * 10.0) * sw.energy.0;
                    estimate +=
                        y_bar(sw.lambda * 10.0) * sw.energy.0 / (pdf.0 + 0.01) / wavelength_pdf;
                    estimate2 +=
                        y_bar(sw.lambda * 10.0) * sw.energy.0 / pdf_for_result / wavelength_pdf;
                    let (px, py) = (
                        (uv.0 * width as f32) as usize,
                        (uv.1 * height as f32) as usize,
                    );

                    // film.buffer[px + width * py] += XYZColor::from(sw) / (pdf.0 + 0.01) / wavelength_pdf;
                    film.buffer[px + width * py] +=
                        XYZColor::from(sw) / (pdf_for_result) / wavelength_pdf;
                }

                if idx % 100 == 0 {
                    pb.add(100);
                    buffer
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(pixel_idx, v)| {
                            let y: usize = pixel_idx / width;
                            let x: usize = pixel_idx - width * y;
                            let [r, g, b, _]: [f32; 4] = converter
                                .transfer_function(
                                    tonemapper.map(&film, (x as usize, y as usize)),
                                    false,
                                )
                                .into();
                            *v =
                                rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
                        });
                    window.update_with_buffer(&buffer, width, height).unwrap();
                }
            }
            println!(
                "\n\nestimate is {}, estimate2 = {}",
                estimate / limit as f32,
                estimate2 / limit as f32
            );

            converter.write_to_files(
                &film,
                Box::new(tonemapper),
                "env_map_sampling_test.exr",
                "env_map_sampling_test.png",
            );
        }
    }

    #[test]
    fn test_env_direct_access() {
        let world = construct_world(PathBuf::from("data/scenes/hdri_test_2.toml")).unwrap();
        let env = &world.environment;

        let wavelength_range = BOUNDED_VISIBLE_RANGE;

        // let mut sum = 0.0;
        let width = 512;
        let height = 256;
        let limit = width * height;
        let deterministic = true;
        let mut estimate = 0.0;
        let mut pb = ProgressBar::new(limit as u64);

        let mut window = Window::new(
            "Preview",
            width,
            height,
            WindowOptions {
                scale: Scale::X2,
                ..WindowOptions::default()
            },
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });
        let mut buffer = vec![0u32; limit];
        let mut film = Film::new(width, height, XYZColor::BLACK);
        let converter = Converter::sRGB;

        let mut tonemapper = Clamp::new(0.0);

        window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

        for idx in 0..limit {
            if !window.is_open() {
                println!("window closed, stopping test");
                break;
            }

            let uv = if deterministic {
                let u = (idx % width) as f32 / width as f32;
                let v = (idx / width) as f32 / height as f32;

                (u, v)
            } else {
                let s = Sample2D::new_random_sample();
                (s.x, s.y)
            };

            let pdf = env.pdf_for(uv);
            for _ in 0..4 {
                let wavelength_sample = if false {
                    // pick a constant wavelength to reduce variance
                    Sample1D::new(0.35)
                } else {
                    Sample1D::new_random_sample()
                };
                let (mut sw, wavelength_pdf) = (
                    SingleWavelength::new_from_range(wavelength_sample.x, wavelength_range),
                    1.0 / wavelength_range.span(),
                );

                sw.energy.0 = env.emission(uv, sw.lambda).0;

                // sum += y_bar(sw.lambda * 10.0) * sw.energy.0;
                estimate += y_bar(sw.lambda * 10.0) * sw.energy.0 / wavelength_pdf;
                let (px, py) = (
                    (uv.0 * width as f32) as usize,
                    (uv.1 * height as f32) as usize,
                );
                film.buffer[px + width * py] +=
                    XYZColor::from(sw) / wavelength_pdf / (pdf.0 + 0.01);
            }

            if idx % 100 == 0 {
                pb.add(100);
                buffer
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(pixel_idx, v)| {
                        let y: usize = pixel_idx / width;
                        let x: usize = pixel_idx - width * y;
                        let [r, g, b, _]: [f32; 4] = converter
                            .transfer_function(
                                tonemapper.map(&film, (x as usize, y as usize)),
                                false,
                            )
                            .into();
                        *v = rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
                    });
                window.update_with_buffer(&buffer, width, height).unwrap();
            }
        }
        println!("\n\nestimate is {}", estimate / limit as f32);
    }

    #[test]
    fn test_2d_importance_sampling() {
        // func is e^(-x^2-y^2)
        let func = |x: f32, y: f32| (-(x * x + y * y)).exp();
        let sample_transform = |x: f32| 4.0 * (x - 0.5);

        // jacobian comes from the sample transform. offsets don't affect the jacobian, but scaling factors do.
        let sample_jacobian = 16.0;

        let resolution = 100;
        let bounds = Bounds1D::new(0.0, 1.0);

        let mut marginal_distribution_data = Vec::new();
        let mut cdfs = Vec::new();
        for y_i in 0..resolution {
            let y_f = y_i as f32 / resolution as f32;
            let row_cdf = Curve::from_function(
                |x| func(sample_transform(x), sample_transform(y_f)),
                resolution,
                bounds,
                InterpolationMode::Linear,
            )
            .to_cdf(bounds, resolution);

            marginal_distribution_data.push(row_cdf.pdf_integral);
            println!("{:?}", row_cdf.pdf_integral);
            cdfs.push(row_cdf);
        }

        // let total = marginal_distribution_data.iter().sum::<f32>();
        // marginal_distribution_data
        //     .iter_mut()
        //     .for_each(|e| *e /= total);

        let importance_map = ImportanceMap::Baked {
            data: cdfs,
            marginal_cdf: Curve::Linear {
                signal: marginal_distribution_data,
                bounds,
                mode: InterpolationMode::Linear,
            }
            .to_cdf(bounds, resolution),
            vertical_resolution: resolution,
            horizontal_resolution: resolution,
            luminance_curve: crate::curves::cie_e(1.0),
        };

        let width = 512;
        let height = 512;
        let limit = width * height;
        let deterministic = false;
        // let limit = 100000;
        let mut estimate = 0.0;
        let mut pb = ProgressBar::new(limit as u64);

        let mut window = Window::new(
            "Preview",
            width,
            height,
            WindowOptions {
                scale: Scale::X2,
                ..WindowOptions::default()
            },
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });
        let mut buffer = vec![0u32; limit];
        let mut film = Film::new(width, height, XYZColor::BLACK);
        let converter = Converter::sRGB;

        let mut tonemapper = Clamp::new(0.0);

        window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

        for idx in 0..limit {
            if !window.is_open() {
                println!("window closed, stopping test");
                break;
            }

            let sample = Sample2D::new_random_sample();
            let x_float = ((idx % width) as f32 + sample.x) / width as f32;
            let y_float = ((idx / width) as f32 + sample.y) / height as f32;

            let sample = if deterministic {
                Sample2D::new(x_float, y_float)
            } else {
                sample
            };

            // let uv = (sample.x, sample.y);
            // let pdf = 1.0;

            let (uv, pdf) = importance_map.sample_uv(sample);
            // println!("{} {}", uv.0, uv.1);
            let pdf = (pdf.0 * pdf.1).0 / sample_jacobian;
            // let uv = (uv.0 / 4.0 + 0.5, uv.1 / 4.0 + 0.5);

            // estimate of env map luminance will have unacceptable bias depending on the actual size of the env map texture.
            // need to downsample to retain size information in importance map so that the pdf can be adjusted.

            let (mut sw, _) = (
                SingleWavelength::new_from_range(0.5, BOUNDED_VISIBLE_RANGE),
                1.0,
            );

            sw.energy.0 = func(sample_transform(uv.0), sample_transform(uv.1));

            // sum += y_bar(sw.lambda * 10.0) * sw.energy.0;
            estimate += sw.energy.0 / pdf / limit as f32;
            let (px, py) = (
                (uv.0 * width as f32) as usize,
                (uv.1 * height as f32) as usize,
            );

            // film.buffer[px + width * py] += XYZColor::from(sw) / (pdf.0 + 0.01) / wavelength_pdf;
            film.buffer[px + width * py] += XYZColor::new(1.0, 1.0, 1.0) * sw.energy.0 / pdf;

            if idx % 100 == 0 {
                println!();
                println!("{}", estimate * limit as f32 / idx as f32);
                pb.add(100);
                buffer
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(pixel_idx, v)| {
                        let y: usize = pixel_idx / width;
                        let x: usize = pixel_idx - width * y;
                        let [r, g, b, _]: [f32; 4] = converter
                            .transfer_function(
                                tonemapper.map(&film, (x as usize, y as usize)),
                                false,
                            )
                            .into();
                        *v = rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
                    });
                window.update_with_buffer(&buffer, width, height).unwrap();
            }
        }
        let true_value = 3.11227031972f64;
        let err = (estimate - true_value as f32).abs() / estimate;
        println!(
            "\n\nestimate is {}, true value is {}, error factor is {}",
            estimate, true_value, err
        );
        assert!(err < 0.0001);
    }
}
