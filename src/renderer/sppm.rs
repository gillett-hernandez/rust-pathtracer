use super::{output_film, Film, Renderer};

use crate::camera::Camera;
use crate::config::*;
use crate::integrator::*;
use crate::math::*;
use crate::profile::Profile;
use crate::world::World;

use std::collections::HashMap;
// use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// use crossbeam::channel::{bounded};
use pbr::ProgressBar;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub struct SPPMRenderer {}

impl SPPMRenderer {
    pub fn new() -> SPPMRenderer {
        // TODO rework this, and make something that actually gives a valid render.
        // implement passes over wavelength rather than doing wavelength mollification.
        // importance sample wavelength from an inverse transform sample of <randomly selected light spectra> x <camera luminance curve>
        unimplemented!();
        SPPMRenderer {}
    }

    pub fn render_sampled(
        mut integrator: SPPMIntegrator,
        settings: &RenderSettings,
        _camera: &Camera,
    ) -> Film<XYZColor> {
        let (width, height) = (settings.resolution.width, settings.resolution.height);
        println!(
            "starting sppm render with film resolution {}x{}",
            width, height
        );
        let min_camera_rays = width * height * settings.min_samples as usize;
        println!(
            "minimum samples per pixel: {}",
            settings.min_samples as usize
        );
        println!("minimum total samples: {}", min_camera_rays);

        let now = Instant::now();

        let mut film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);

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

        let mut stats: Profile = Profile::default();
        let mut presampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let mut preprofile = Profile::default();

        for s in 0..settings.min_samples {
            integrator.preprocess(&mut presampler, &vec![settings.clone()], &mut preprofile);

            let clone2 = pixel_count.clone();
            let local_stats: Profile = film
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
                        (x as f32 + sample.x) / (settings.resolution.width as f32),
                        (y as f32 + sample.y) / (settings.resolution.height as f32),
                    );
                    temp_color +=
                        integrator.color(&mut sampler, (camera_uv, 0), s as usize, &mut profile);
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
                    *pixel_ref = temp_color / (settings.min_samples as f32);
                    // }
                    profile
                })
                .reduce(|| Profile::default(), |a, b| a.combine(b));
            stats = stats.combine(local_stats);
        }
        if let Err(panic) = thread.join() {
            println!(
                "progress bar incrememnting thread threw an error {:?}",
                panic
            );
        }

        println!("");
        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;
        println!("took {}s", elapsed);
        stats.pretty_print(elapsed, settings.threads.unwrap() as usize);
        film
    }
}

impl Renderer for SPPMRenderer {
    fn render(&self, mut world: World, cameras: Vec<Camera>, config: &Config) {
        // bin the render settings into bins corresponding to what integrator they need.

        let mut bundled_cameras: Vec<Camera> = Vec::new();
        // let mut films: Vec<(RenderSettings, Film<XYZColor>)> = Vec::new();
        let mut sampled_renders: Vec<(IntegratorType, RenderSettings)> = Vec::new();
        let mut splatting_renders_and_cameras: HashMap<
            IntegratorType,
            Vec<(RenderSettings, Camera)>,
        > = HashMap::new();

        // phase 1, gather and sort what renders need to be done
        for (_render_id, render_settings) in config.render_settings.iter().enumerate() {
            let camera_id = render_settings.camera_id;

            let (width, height) = (
                render_settings.resolution.width,
                render_settings.resolution.height,
            );
            let aspect_ratio = width as f32 / height as f32;

            // copy camera and modify its aspect ratio (so that uv splatting works correctly)
            let copied_camera = cameras[camera_id].with_aspect_ratio(aspect_ratio);

            let integrator_type: IntegratorType = IntegratorType::from(render_settings.integrator);

            match integrator_type {
                IntegratorType::PathTracing => {
                    let mut updated_render_settings = render_settings.clone();
                    updated_render_settings.camera_id = camera_id;
                    bundled_cameras.push(copied_camera);
                    sampled_renders.push((IntegratorType::PathTracing, updated_render_settings));
                }
                IntegratorType::SPPM => {
                    let mut updated_render_settings = render_settings.clone();
                    updated_render_settings.camera_id = camera_id;
                    bundled_cameras.push(copied_camera);
                    sampled_renders.push((IntegratorType::SPPM, updated_render_settings));
                }
                t if splatting_renders_and_cameras.contains_key(&t) => {
                    // then determine new camera id
                    let list = splatting_renders_and_cameras.get_mut(&t).unwrap();
                    let mut updated_render_settings = render_settings.clone();
                    updated_render_settings.camera_id = camera_id;

                    list.push((updated_render_settings, copied_camera))
                }
                _ => {}
            }
        }
        // phase 2, for renders that don't require a splatted render, do them first, and output results as soon as they're finished

        for (integrator_type, render_settings) in sampled_renders.iter() {
            match integrator_type {
                IntegratorType::SPPM => {
                    world.assign_cameras(vec![cameras[render_settings.camera_id].clone()], false);
                    let arc_world = Arc::new(world.clone());
                    match Integrator::from_settings_and_world(
                        arc_world.clone(),
                        IntegratorType::SPPM,
                        &bundled_cameras,
                        render_settings,
                    ) {
                        Some(Integrator::SPPM(integrator)) => {
                            println!("rendering with sppm integrator");
                            let (render_settings, film) = (
                                render_settings.clone(),
                                SPPMRenderer::render_sampled(
                                    integrator,
                                    render_settings,
                                    &cameras[render_settings.camera_id],
                                ),
                            );
                            output_film(&render_settings, &film);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
}
