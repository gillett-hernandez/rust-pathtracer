use super::{output_film, Film, Renderer};

use crate::config::Config;
use crate::integrator::*;
use crate::math::*;
use crate::{camera::Camera, config::RendererType};

use crate::profile::Profile;
use crate::tonemap::{sRGB, Tonemapper};
use crate::world::World;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// use crossbeam::channel::unbounded;

use pbr::ProgressBar;

use minifb::{Key, Scale, Window, WindowOptions};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub struct PreviewRenderer {}

impl PreviewRenderer {
    pub fn new() -> Self {
        PreviewRenderer {}
    }
}
fn rgb_to_u32(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

impl Renderer for PreviewRenderer {
    fn render(&self, mut world: World, cameras: Vec<Camera>, config: &Config) {
        use crate::config::Resolution;

        if let RendererType::Preview {
            selected_preview_film_id,
        } = config.renderer
        {
            let mut films: Vec<Film<XYZColor>> = Vec::new();
            for render_settings in config.render_settings.iter() {
                let Resolution { width, height } = render_settings.resolution;
                films.push(Film::new(width, height, XYZColor::BLACK));
            }
            println!("num cameras {}", cameras.len());
            println!("num cameras {}", world.cameras.len());

            let film_idx = selected_preview_film_id;
            let render_settings = config.render_settings[film_idx].clone();

            world.assign_cameras(
                vec![cameras[render_settings.camera_id.unwrap() as usize].clone()],
                false,
            );
            let arc_world = Arc::new(world.clone());

            let Resolution { width, height } = render_settings.resolution;

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
            // Limit to max ~60 fps update rate
            window.limit_update_rate(Some(std::time::Duration::from_micros(16666)));
            let mut buffer = vec![0u32; width * height];
            if let Some(Integrator::PathTracing(integrator)) = Integrator::from_settings_and_world(
                arc_world.clone(),
                IntegratorType::PathTracing,
                &cameras,
                &render_settings,
            ) {
                let min_camera_rays = width * height * render_settings.min_samples as usize;
                println!("minimum total samples: {}", min_camera_rays);

                let mut pb = ProgressBar::new((width * height) as u64);

                let total_pixels = width * height;

                let pixel_count = Arc::new(AtomicUsize::new(0));
                let clone1 = pixel_count.clone();
                let thread = thread::spawn(move || {
                    let mut local_index = 0;
                    while local_index < total_pixels {
                        let pixels_to_increment = clone1.load(Ordering::Relaxed) - local_index;
                        pb.add(pixels_to_increment as u64);
                        local_index += pixels_to_increment;

                        thread::sleep(Duration::from_millis(250));
                    }
                });

                let clone2 = pixel_count.clone();
                for s in 0..render_settings.min_samples {
                    if !window.is_open() || window.is_key_down(Key::Escape) {
                        break;
                    }
                    let now = Instant::now();
                    let stats: Profile = (&mut films[film_idx])
                        .buffer
                        .par_iter_mut()
                        .enumerate()
                        .map(|(pixel_index, pixel_ref)| {
                            let mut profile = Profile::default();
                            // let clone = pixel_count.clone();
                            let y: usize = pixel_index / width;
                            let x: usize = pixel_index - width * y;
                            // gen ray for pixel x, y
                            // let r: Ray = Ray::new(Point3::ZERO, Vec3::X);
                            // let mut temp_color = RGBColor::BLACK;
                            let mut temp_color = XYZColor::BLACK;
                            // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
                            let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                            // idea: use SPD::Tabulated to collect all the data for a single pixel as a SPD, then convert that whole thing to XYZ.

                            let sample = sampler.draw_2d();

                            let camera_uv = (
                                (x as f32 + sample.x) / (render_settings.resolution.width as f32),
                                (y as f32 + sample.y) / (render_settings.resolution.height as f32),
                            );
                            temp_color += XYZColor::from(integrator.color(
                                &mut sampler,
                                (camera_uv, 0),
                                s as usize,
                                &mut profile,
                            ));
                            // temp_color += RGBColor::from(integrator.color(&mut sampler, r));
                            debug_assert!(
                                temp_color.0.is_finite().all(),
                                "{:?} resulted in {:?}",
                                camera_uv,
                                temp_color
                            );

                            clone2.fetch_add(1, Ordering::Relaxed);
                            // if pixel_index % output_divisor == 0 {
                            //     let stdout = std::io::stdout();
                            //     let mut handle = stdout.lock();
                            //     handle.write_all(b".").unwrap();
                            //     std::io::stdout().flush().expect("some error message")
                            // }
                            // pb.inc();
                            // unsafe {
                            *pixel_ref += temp_color / (render_settings.min_samples as f32);
                            // }
                            profile
                        })
                        .reduce(|| Profile::default(), |a, b| a.combine(b));
                    println!("");
                    let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;
                    println!("took {}s", elapsed);
                    stats.pretty_print(elapsed, render_settings.threads.unwrap() as usize);
                    let srgb_tonemapper =
                        sRGB::new(&films[film_idx], render_settings.exposure.unwrap_or(1.0));
                    buffer
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(pixel_idx, v)| {
                            let y: usize = pixel_idx / width;
                            let x: usize = pixel_idx - width * y;
                            let (mapped, _linear) = srgb_tonemapper.map(&films[film_idx], (x, y));
                            let [r, g, b, _]: [f32; 4] = mapped.into();
                            *v =
                                rgb_to_u32((255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8);
                        });
                    window.update_with_buffer(&buffer, width, height).unwrap();
                }
                if let Err(panic) = thread.join() {
                    println!(
                        "progress bar incrememnting thread threw an error {:?}",
                        panic
                    );
                }
                output_film(&render_settings, &films[film_idx]);
            }
        }
    }
}
