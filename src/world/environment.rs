use std::sync::Arc;

use math::spectral::Op;
use parking_lot::Mutex;

use crate::texture::TexStack;
use crate::{math::*, rgb_to_u32};

use minifb::{Scale, Window, WindowOptions};
use pbr::ProgressBar;

use rayon::iter::ParallelIterator;
use rayon::prelude::*;

#[derive(Clone)]
pub struct ImportanceMap {
    data: Vec<CDF>,
    vertical_cdf: CDF,
    // wavelength_bounds: Bounds1D,
}

impl ImportanceMap {
    pub fn new(
        texture_stack: &TexStack,
        vertical_resolution: usize,
        horizontal_resolution: usize,
        luminance_curve: SPD,
        wavelength_bounds: Bounds1D,
    ) -> Self {
        let num_samples_for_texel_spectra = 100;
        let mut data = Vec::new();
        let mut v_cdf = Vec::new();

        let mut total_luminance = 0.0;

        // let mut machine = SPD::Machine {
        //     seed: 1.0,
        //     list: vec![(Op::Mul, luminance_curve), (Op::Mul, SPD::Const(0.0))],
        // };

        let (mut window, mut buffer) = if cfg!(feature = "visualize") {
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
            )
        } else {
            (None, vec![])
        };

        println!("generating importance map");
        let pb = Arc::new(Mutex::new(ProgressBar::new(
            (vertical_resolution * horizontal_resolution) as u64,
        )));

        let mut rows: Vec<(CDF, f32)> = (0..vertical_resolution)
            .into_par_iter()
            .map(|row| -> (CDF, f32) {
                let mut spd = Vec::new();
                let mut cdf = Vec::new();
                let mut row_luminance = 0.0;
                for column in 0..horizontal_resolution {
                    let uv = (
                        row as f32 / vertical_resolution as f32,
                        column as f32 / horizontal_resolution as f32,
                    );
                    let machine = SPD::Machine {
                        seed: 1.0,
                        list: vec![
                            (Op::Mul, luminance_curve.clone()),
                            (Op::Mul, texture_stack.curve_at(uv)),
                        ],
                    };
                    let texel_luminance = machine.evaluate_integral(
                        wavelength_bounds,
                        wavelength_bounds.span() / num_samples_for_texel_spectra as f32,
                        false,
                    );
                    row_luminance += texel_luminance;
                    spd.push(texel_luminance);
                    cdf.push(row_luminance);
                }
                // normalize pdf (and cdf?)
                spd.iter_mut().for_each(|e| {
                    *e /= row_luminance;
                });
                cdf.iter_mut().for_each(|e| {
                    *e /= row_luminance;
                });
                pb.lock().add(horizontal_resolution as u64);
                (
                    CDF {
                        pdf: SPD::Linear {
                            signal: spd,
                            bounds: Bounds1D::new(0.0, 1.0),
                            mode: InterpolationMode::Linear,
                        },
                        cdf: SPD::Linear {
                            signal: cdf,
                            bounds: Bounds1D::new(0.0, 1.0),
                            mode: InterpolationMode::Linear,
                        },
                        cdf_integral: row_luminance,
                    },
                    row_luminance,
                )
            })
            .collect();
        for (row, (cdf, row_luminance)) in rows.drain(..).enumerate() {
            // let  = rows[row];
            if cfg!(feature = "visualize") {
                for column in 0..horizontal_resolution {
                    let rgb = (cdf
                        .cdf
                        .evaluate_power(column as f32 / horizontal_resolution as f32)
                        / row_luminance)
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
            v_cdf.push(total_luminance);

            if let Some(window) = &mut window {
                window
                    .update_with_buffer(&buffer, horizontal_resolution, vertical_resolution)
                    .unwrap();
            }
        }
        pb.lock().finish();

        if let Some(window) = &mut window {
            for row in 0..vertical_resolution {
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

        v_cdf.iter_mut().for_each(|e| *e /= total_luminance);
        let vertical_cdf = SPD::Linear {
            signal: v_cdf,
            bounds: Bounds1D::new(0.0, 1.0),
            mode: InterpolationMode::Cubic,
        }
        .into();

        Self {
            data,
            vertical_cdf,
            // wavelength_bounds,
        }
    }
}

#[derive(Clone)]
pub enum EnvironmentMap {
    Constant {
        color: CDF,
        strength: f32,
    },
    Sun {
        color: CDF,
        strength: f32,
        angular_diameter: f32,
        sun_direction: Vec3,
    },
    HDRi {
        texture: TexStack,
        importance_map: Option<ImportanceMap>,
        rotation: Transform3,
        strength: f32,
    },
}

impl EnvironmentMap {
    // pub const fn new(color: SPD, strength: f32) -> Self {
    //     EnvironmentMap { color, strength }
    // }

    // currently unused
    // sample the spectral distribution at a env map UV
    // used when a camera ray hits an environment map without ever having been assigned a wavelength.
    // would happen when a camera ray hits an env map without bouncing on anything wavelength dependent
    // assuming that the wavelength shouldn't have already been sampled based on the camera's spectral sensitivity
    // pub fn _sample_spd(
    //     &self,
    //     _uv: (f32, f32),
    //     wavelength_range: Bounds1D,
    //     wavelength_sample: Sample1D,
    // ) -> Option<(SingleWavelength, PDF)> {
    //     // later use uv for texture accessing
    //     let (mut sw, pdf) = self
    //         .color
    //         .sample_power_and_pdf(wavelength_range, wavelength_sample);
    //     sw.energy *= self.strength;
    //     Some((sw, pdf))
    // }

    // evaluate env map given a uv and wavelength
    // used when a camera ray with a given wavelength intersects the environment map
    #[allow(unused_variables)]
    pub fn emission(&self, uv: (f32, f32), lambda: f32) -> SingleEnergy {
        // evaluate emission at uv coordinate and wavelength
        match self {
            EnvironmentMap::Constant { color, strength } => {
                debug_assert!(lambda > 0.0);
                // SingleEnergy::new(self.color.evaluate_power(lambda))
                SingleEnergy::new(color.evaluate_power(lambda) * strength)
            }
            EnvironmentMap::Sun {
                color,
                strength,
                angular_diameter,
                sun_direction,
            } => {
                let direction = uv_to_direction(uv);
                let cos = *sun_direction * direction;
                let sin = (1.0 - cos * cos).sqrt();
                if sin.abs() < (*angular_diameter / 2.0).sin() && cos > 0.0 {
                    // within solid angle
                    SingleEnergy::new(color.evaluate_power(lambda) * *strength)
                } else {
                    SingleEnergy::ZERO
                }
            }
            EnvironmentMap::HDRi {
                texture,
                rotation,
                importance_map,
                strength,
            } => {
                let direction = uv_to_direction(uv);
                let new_direction = rotation.to_local(direction);
                let uv = direction_to_uv(new_direction);
                let result = texture.eval_at(lambda, uv) * strength;
                result.into()
            }
        }
    }

    // sample a ray and wavelength based on env map CDF
    pub fn sample_emission(
        &self,
        world_radius: f32,
        world_center: Point3,
        position_sample: Sample2D,
        direction_sample: Sample2D,
        wavelength_range: Bounds1D,
        wavelength_sample: Sample1D,
    ) -> (Ray, SingleWavelength, PDF, PDF) {
        // sample env map cdf to get light ray, based on env map strength
        match self {
            EnvironmentMap::Constant { color, strength } => {
                let (mut sw, pdf) = color.sample_power_and_pdf(wavelength_range, wavelength_sample);
                sw.energy *= *strength;
                let random_direction = random_on_unit_sphere(direction_sample);
                let frame = TangentFrame::from_normal(random_direction);
                let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
                let point = world_center
                    + -random_direction * world_radius
                    + frame.to_world(&random_on_normal_disk);

                (
                    Ray::new(point, random_direction),
                    sw,
                    // pdf * 1.0 / (4.0 * PI), // solid angle pdf w/ wavelength sample incorporated
                    PDF::from(1.0 / (4.0 * PI)), // solid angle pdf
                    pdf,                         // wavelength pdf
                )
            }
            EnvironmentMap::Sun {
                color,
                strength,
                angular_diameter: _,
                sun_direction: _,
            } => {
                let (mut sw, wavelength_pdf) =
                    color.sample_power_and_pdf(wavelength_range, wavelength_sample);
                sw.energy *= *strength;
                let (uv, directional_pdf) =
                    self.sample_env_uv_given_wavelength(direction_sample, sw.lambda);

                let direction = uv_to_direction(uv);
                let frame = TangentFrame::from_normal(direction);
                let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
                let point = world_center
                    + direction * world_radius
                    + frame.to_world(&random_on_normal_disk);

                (
                    Ray::new(point, -direction),
                    sw,
                    directional_pdf,
                    wavelength_pdf,
                )
            }
            EnvironmentMap::HDRi {
                texture,
                rotation: _,
                importance_map: _,
                strength,
            } => {
                // let (mut sw, wavelength_pdf) =
                //     color.sample_power_and_pdf(wavelength_range, wavelength_sample);
                // sw.energy *= *strength;
                let (uv, directional_pdf) = self.sample_env_uv(direction_sample);
                let (mut sw, wavelength_pdf) = (
                    SingleWavelength::new_from_range(wavelength_sample.x, wavelength_range),
                    1.0 / wavelength_range.span(),
                );
                sw.energy.0 = texture.eval_at(sw.lambda, uv) * strength;

                let direction = uv_to_direction(uv);
                let frame = TangentFrame::from_normal(direction);
                let random_on_normal_disk = world_radius * random_in_unit_disk(position_sample);
                let point = world_center
                    + direction * world_radius
                    + frame.to_world(&random_on_normal_disk);

                (
                    Ray::new(point, -direction),
                    sw,
                    directional_pdf,
                    wavelength_pdf.into(),
                )
            }
        }
    }

    pub fn sample_direction_given_wavelength(&self, sample: Sample2D, lambda: f32) -> (Vec3, PDF) {
        let (uv, pdf) = self.sample_env_uv_given_wavelength(sample, lambda);
        let direction = uv_to_direction(uv);

        (direction, pdf)
    }

    pub fn sample_direction_and_wavelength(
        &self,
        _sample: Sample2D,
        _wavelength_range: Bounds1D,
        _wavelength_sample: Sample1D,
    ) -> (Vec3, PDF) {
        unimplemented!()
    }

    // sample env UV given a wavelength, based on env CDF for a specific wavelength. might be hard to evaluate, or nearly impossible.
    // would be used when sampling the environment from an eye path, such as in PT or BDPT, given a wavelength
    pub fn sample_env_uv_given_wavelength(
        &self,
        sample: Sample2D,
        _lambda: f32,
    ) -> ((f32, f32), PDF) {
        match self {
            EnvironmentMap::Constant { .. } => self.sample_env_uv(sample),
            EnvironmentMap::Sun { .. } => self.sample_env_uv(sample),
            EnvironmentMap::HDRi { .. } => {
                // do stuff
                self.sample_env_uv(sample)
            }
        }

        // however because that's unimplemented for now, lets just return `sample_env_uv`
    }

    // sample env UV, based on env luminosity CDF (w/o prescribed wavelength)
    pub fn sample_env_uv(&self, sample: Sample2D) -> ((f32, f32), PDF) {
        // samples env CDF to find bright luminosity spikes. returns UV of those spots.
        // CDF for this situation can be stored as the Y values of the XYZ representation, as a greyscale image potentially.
        // consider summed area table as well.
        match self {
            EnvironmentMap::Constant { .. } => ((sample.x, sample.y), PDF::from(1.0 / (4.0 * PI))),
            EnvironmentMap::Sun {
                angular_diameter,
                sun_direction,
                ..
            } => {
                let local_wo =
                    Vec3::Z + (*angular_diameter / 2.0).sin() * random_in_unit_disk(sample);
                let sun_direction = *sun_direction;
                let frame = TangentFrame::from_normal(sun_direction);
                let direction = frame.to_world(&local_wo);
                (
                    direction_to_uv(direction.normalized()),
                    PDF::from(1.0 / (2.0 * PI * (1.0 - *angular_diameter))),
                    // 1.0.into()
                )
            }
            EnvironmentMap::HDRi {
                rotation,
                importance_map,
                ..
            } => {
                if let Some(importance_map) = importance_map {
                    // inverse transform sample the vertical cdf

                    let (
                        SingleWavelength {
                            lambda: u,
                            energy: _,
                        },
                        row_pdf,
                    ) = importance_map.vertical_cdf.sample_power_and_pdf(
                        // cdf inverse transform sample should still work even though the wavelength bounds are 0..1
                        Bounds1D::new(0.0, 1.0),
                        Sample1D::new(sample.y),
                    );
                    // inverse transform sample the selected horizontal cdf
                    let (
                        SingleWavelength {
                            lambda: v,
                            energy: _,
                        },
                        column_pdf,
                    ) = importance_map.data[(u * importance_map.data.len() as f32) as usize]
                        .sample_power_and_pdf(
                            // cdf inverse transform sample should still work even though the wavelength bounds are 0..1
                            Bounds1D::new(0.0, 1.0),
                            Sample1D::new(sample.x),
                        );
                    assert!(u < 1.0, "{}", u);
                    assert!(u >= 0.0, "{}", u);
                    assert!(v < 1.0, "{}", v);
                    assert!(v >= 0.0, "{}", v);
                    let local_wo = uv_to_direction((u, v));
                    let new_wo = rotation.to_world(local_wo);
                    let uv = direction_to_uv(new_wo);
                    (uv, PDF::from(row_pdf * column_pdf))
                    // ((sample.x, sample.y), PDF::from(1.0 / (4.0 * PI)))
                } else {
                    ((sample.x, sample.y), PDF::from(1.0 / (4.0 * PI)))
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use math::spectral::{y_bar, BOUNDED_VISIBLE_RANGE};

    use super::*;
    use crate::curves;
    #[test]
    fn test_sample_emission() {
        let env_map = EnvironmentMap::Constant {
            color: curves::blackbody(5500.0, 40.0).into(),
            strength: 1.0,
        };
        let (ray, sw, pdf, _lambda_pdf) = env_map.sample_emission(
            1.0,
            Point3::ORIGIN,
            Sample2D::new_random_sample(),
            Sample2D::new_random_sample(),
            curves::EXTENDED_VISIBLE_RANGE,
            Sample1D::new_random_sample(),
        );
        println!("{:?} {:?} {:?}", ray, sw, pdf);
        let Ray {
            origin,
            direction,
            time: _,
            tmax: _,
        } = ray;

        let dir_toward_world_origin = Point3::ORIGIN - origin;
        let dot = dir_toward_world_origin * direction;
        println!("{}", dot);
    }

    #[test]
    fn test_sample_emission_sun() {
        let env_map = EnvironmentMap::Sun {
            color: curves::blackbody(5500.0, 40.0).into(),
            strength: 1.0,
            angular_diameter: 0.1,
            sun_direction: Vec3::Z,
        };
        let (ray, sw, pdf, _lambda_pdf) = env_map.sample_emission(
            1.0,
            Point3::ORIGIN,
            Sample2D::new_random_sample(),
            Sample2D::new_random_sample(),
            curves::EXTENDED_VISIBLE_RANGE,
            Sample1D::new_random_sample(),
        );
        println!("{:?} {:?} {:?}", ray, sw, pdf);
        let Ray {
            origin,
            direction,
            time: _,
            tmax: _,
        } = ray;

        let dir_toward_world_origin = Point3::ORIGIN - origin;
        let dot = dir_toward_world_origin * direction;
        println!("{}", dot);
    }

    #[test]
    fn test_uv_to_direction_and_back() {
        let direction = Vec3::new(1.0, 1.0, 1.0).normalized();
        println!("{:?}", direction);
        let uv = direction_to_uv(direction);
        println!("{:?}", uv);
        let direction_again = uv_to_direction(uv);
        println!("{:?}", direction_again);
    }

    #[test]
    fn test_sample_env_map() {
        let env_map = EnvironmentMap::Sun {
            color: curves::blackbody(5500.0, 40.0).into(),
            strength: 1.0,
            angular_diameter: 0.1,
            sun_direction: Vec3::Z,
        };

        env_map.sample_direction_given_wavelength(Sample2D::new_random_sample(), 500.0);
    }
    use crate::parsing::construct_world;
    use crate::renderer::Film;
    use crate::tonemap::{sRGB, Tonemapper};
    #[test]
    fn test_env_importance_sampling() {
        let world = construct_world("data/scenes/hdri_test.toml");
        let env = &world.environment;
        if let EnvironmentMap::HDRi {
            importance_map,
            texture,
            strength,
            ..
        } = env
        {
            if importance_map.is_some() {
                println!("got importance map, now do some math on it to see if it works");
            } else {
                println!("testing env map without importance map, now do some math on it to see if it works");
            }

            let wavelength_range = BOUNDED_VISIBLE_RANGE;

            let limit = 100000;

            // let mut sum = 0.0;
            let mut estimate = 0.0;
            let mut pb = ProgressBar::new(limit as u64);

            let width = 512;
            let height = 512;
            let mut window = Window::new(
                "Preview",
                width,
                height,
                WindowOptions {
                    scale: Scale::X1,
                    ..WindowOptions::default()
                },
            )
            .unwrap_or_else(|e| {
                panic!("{}", e);
            });
            let mut buffer = vec![0u32; width * height];
            let mut film = Film::new(width, height, XYZColor::BLACK);

            window.limit_update_rate(Some(std::time::Duration::from_micros(6944)));

            for idx in 0..limit {
                // env.sample_emission(
                //     world.get_world_radius(),
                //     world.get_center(),
                //     Sample2D::new_random_sample(),
                //     Sample2D::new_random_sample(),
                //     BOUNDED_VISIBLE_RANGE,
                //     Sample1D::new_random_sample(),
                // );
                let (uv, pdf) = env.sample_env_uv(Sample2D::new_random_sample());

                let wavelength_sample = Sample1D::new_random_sample();
                let (mut sw, wavelength_pdf) = (
                    SingleWavelength::new_from_range(wavelength_sample.x, wavelength_range),
                    1.0 / wavelength_range.span(),
                );

                sw.energy.0 = texture.eval_at(sw.lambda, uv) * strength;

                // sum += y_bar(sw.lambda * 10.0) * sw.energy.0;
                estimate += y_bar(sw.lambda * 10.0) * sw.energy.0 / pdf.0 / wavelength_pdf;
                let (px, py) = (
                    (uv.0 * width as f32) as usize,
                    (uv.1 * height as f32) as usize,
                );
                film.buffer[px + width * py] += XYZColor::from(sw) / pdf.0 / wavelength_pdf;

                pb.add(1);
                if idx % 100 == 0 {
                    let srgb_tonemapper = sRGB::new(&film, 1.0, false);
                    buffer
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(pixel_idx, v)| {
                            let y: usize = pixel_idx / width;
                            let x: usize = pixel_idx - width * y;
                            let (mapped, _linear) = srgb_tonemapper.map(&film, (x, y));
                            let [r, g, b, _]: [f32; 4] = mapped.into();
                            *v =
                                rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
                        });
                    window.update_with_buffer(&buffer, width, height).unwrap();
                }
            }
            println!("\n\nestimate is {}", estimate);
        }
    }
}
