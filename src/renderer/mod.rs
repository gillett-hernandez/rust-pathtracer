mod film;

pub use film::Film;

use crate::camera::{Camera, CameraId};
use crate::config::Config;
use crate::config::RenderSettings;
use crate::integrator::*;
use crate::math::*;
use crate::tonemap::{sRGB, Tonemapper};
use crate::world::World;

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

// fn construct_integrator(
//     settings: &RenderSettings,
//     world: Arc<World>,
// ) -> Box<dyn SamplerIntegrator> {
//     let max_bounces = settings.max_bounces.unwrap_or(1);
//     let russian_roulette = settings.russian_roulette.unwrap_or(true);
//     let light_samples = settings.light_samples.unwrap_or(4);
//     let only_direct = settings.only_direct.unwrap_or(false);
// }

pub struct NaiveRenderer {}

impl NaiveRenderer {
    pub fn new() -> NaiveRenderer {
        NaiveRenderer {}
    }

    pub fn render_sampled<I: SamplerIntegrator>(
        integrator: I,
        settings: &RenderSettings,
        camera: &Camera,
    ) -> Film<XYZColor> {
        let (width, height) = (settings.resolution.width, settings.resolution.height);
        println!("starting render with film resolution {}x{}", width, height);
        let min_camera_rays = width * height * settings.min_samples as usize;
        println!("minimum total samples: {}", min_camera_rays);

        let now = Instant::now();

        let total_camera_rays =
            width * height * (settings.max_samples.unwrap_or(settings.min_samples) as usize);

        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;

        // do stuff with film here

        let mut film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);

        for _ in 0..100 {
            print!("-");
        }
        println!("");
        let output_divisor = (film.width * film.height / 100).max(1);
        film.buffer
            .par_iter_mut()
            // .iter_mut()
            .enumerate()
            .for_each(|(pixel_index, pixel_ref)| {
                let y: usize = pixel_index / width;
                let x: usize = pixel_index - width * y;
                // gen ray for pixel x, y
                // let r: Ray = Ray::new(Point3::ZERO, Vec3::X);
                // let mut temp_color = RGBColor::BLACK;
                let mut temp_color = XYZColor::BLACK;
                let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
                // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                // idea: use SPD::Tabulated to collect all the data for a single pixel as a SPD, then convert that whole thing to XYZ.
                for _s in 0..settings.min_samples {
                    let sample = sampler.draw_2d();
                    let r = camera.get_ray(
                        (x as f32 + sample.x) / (width as f32),
                        (y as f32 + sample.y) / (height as f32),
                    );
                    temp_color += XYZColor::from(integrator.color(&mut sampler, r));
                    // temp_color += RGBColor::from(integrator.color(&mut sampler, r));
                    assert!(
                        temp_color.0.is_finite().all(),
                        "{:?} resulted in {:?}",
                        r,
                        temp_color
                    );
                }
                if pixel_index % output_divisor == 0 {
                    let stdout = std::io::stdout();
                    let mut handle = stdout.lock();
                    handle.write_all(b".").unwrap();
                    std::io::stdout().flush().expect("some error message")
                }
                // unsafe {
                *pixel_ref = temp_color / (settings.min_samples as f32);
                // }
            });
        println!("");
        println!(
            "\ntook {}s at {} rays per second and {} rays per second per thread",
            elapsed,
            (total_camera_rays as f32) / elapsed,
            (total_camera_rays as f32) / elapsed / (settings.threads.unwrap() as f32)
        );
        film
    }
    pub fn render_splatted<I: GenericIntegrator>(
        integrator: I,
        renders: Vec<RenderSettings>,
        cameras: Vec<Camera>,
    ) -> Vec<Film<XYZColor>> {
        // let (width, height) = (settings.resolution.width, settings.resolution.height);
        // let min_camera_rays = width * height * settings.min_samples as usize;

        let now = Instant::now();

        // let total_camera_rays =
        //     width * height * (settings.max_samples.unwrap_or(settings.min_samples) as usize);

        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;

        // do stuff with film here

        let mut total_camera_samples = 0;
        let mut total_pixels = 0;
        let mut films: Vec<(RenderSettings, Film<XYZColor>)> = Vec::new();
        let mut light_films: Vec<Film<XYZColor>> = Vec::new();
        for settings in renders.iter() {
            let (width, height) = (settings.resolution.width, settings.resolution.height);
            println!("starting render with film resolution {}x{}", width, height);
            let pixels = width * height;
            total_pixels += pixels;
            total_camera_samples += pixels * (settings.min_samples as usize);
            let image_film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);
            let light_film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);
            films.push((settings.clone(), image_film));
            light_films.push(light_film);
        }
        println!("total pixels: {}", total_pixels);
        println!("minimum total samples: {}", total_camera_samples);

        for _ in 0..100 {
            print!("-");
        }

        println!("");
        let mut additional_splats: Vec<(Sample, CameraId)> = Vec::new();
        let result: Vec<(Sample, CameraId)> = films
            .par_iter_mut()
            .enumerate()
            .flat_map(
                |(film_number, (settings, film)): (
                    usize,
                    &mut (RenderSettings, Film<XYZColor>),
                )|
                 -> Vec<(Sample, CameraId)> {
                    let output_divisor = (film.width * film.height / 100).max(1);
                    let additional_samples = film
                        .buffer
                        .par_iter_mut()
                        // .iter_mut()
                        .enumerate()
                        .flat_map(|(pixel_index, pixel_ref)| -> Vec<(Sample, CameraId)> {
                            let y: usize = pixel_index / settings.resolution.width;
                            let x: usize = pixel_index - settings.resolution.width * y;
                            let camera = cameras[settings.camera_id.unwrap() as usize];
                            // gen ray for pixel x, y
                            // let r: Ray = Ray::new(Point3::ZERO, Vec3::X);
                            // let mut temp_color = RGBColor::BLACK;
                            let mut temp_color = XYZColor::BLACK;
                            let mut sampler: Box<dyn Sampler> =
                                Box::new(StratifiedSampler::new(20, 20, 10));
                            // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                            // idea: use SPD::Tabulated to collect all the data for a single pixel as a SPD, then convert that whole thing to XYZ.
                            let mut local_additional_splats: Vec<(Sample, CameraId)> = Vec::new();
                            for _s in 0..settings.min_samples {
                                let sample = sampler.draw_2d();
                                let r = camera.get_ray(
                                    (x as f32 + sample.x) / (settings.resolution.width as f32),
                                    (y as f32 + sample.y) / (settings.resolution.height as f32),
                                );
                                temp_color += XYZColor::from(integrator.color(
                                    &mut sampler,
                                    r,
                                    &mut local_additional_splats,
                                ));
                                // temp_color += RGBColor::from(integrator.color(&mut sampler, r));
                                assert!(
                                    temp_color.0.is_finite().all(),
                                    "{:?} resulted in {:?}",
                                    r,
                                    temp_color
                                );
                            }
                            if pixel_index % output_divisor == 0 {
                                let stdout = std::io::stdout();
                                let mut handle = stdout.lock();
                                handle.write_all(b".").unwrap();
                                std::io::stdout().flush().expect("some error message")
                            }
                            // unsafe {
                            *pixel_ref = temp_color / (settings.min_samples as f32);
                            local_additional_splats
                            // }
                        })
                        .collect();
                    additional_samples
                },
            )
            .collect();
        use std::cmp::Ordering;
        // additional_splats.;
        additional_splats.par_sort_unstable_by(|(sample1, camera_id1), (sample2, camera_id2)| {
            camera_id1.0.cmp(&camera_id2.0)
        });

        println!("");
        let maximum_threads = renders
            .iter()
            .max_by_key(|s| s.threads)
            .unwrap()
            .threads
            .unwrap();
        println!(
            "\ntook {}s at {} rays per second and {} rays per second per thread",
            elapsed,
            (total_camera_samples as f32) / elapsed,
            (total_camera_samples as f32) / elapsed / (maximum_threads as f32)
        );

        Vec::new()
    }
}

pub trait Renderer {
    fn render(&self, world: World, cameras: Vec<Camera>, config: &Config);
}

impl Renderer for NaiveRenderer {
    fn render(&self, world: World, cameras: Vec<Camera>, config: &Config) {
        // bin the render settings into bins corresponding to what integrator they need.

        let bundled_cameras: Vec<Camera> = Vec::new();
        let mut films: Vec<(RenderSettings, Film<XYZColor>)> = Vec::new();
        let mut sampled_renders: Vec<(IntegratorType, RenderSettings)> = Vec::new();
        let mut batched_renders_and_cameras: HashMap<
            IntegratorType,
            Vec<(RenderSettings, Camera)>,
        > = HashMap::new();
        batched_renders_and_cameras.insert(IntegratorType::PathTracing, Vec::new());
        batched_renders_and_cameras.insert(IntegratorType::BDPT, Vec::new());

        // phase 1, gather and sort what renders need to be done
        for (_render_id, render_settings) in config.render_settings.iter().enumerate() {
            let camera_id = render_settings.camera_id.unwrap_or(0) as usize;

            let (width, height) = (
                render_settings.resolution.width,
                render_settings.resolution.height,
            );
            let aspect_ratio = width as f32 / height as f32;

            // copy camera and modify its aspect ratio (so that uv splatting works correctly)
            let copied_camera = cameras[camera_id].with_aspect_ratio(aspect_ratio);

            let integrator_type: IntegratorType = IntegratorType::from_string(
                &render_settings
                    .integrator
                    .as_ref()
                    .unwrap_or(&"PT".to_string()),
            );

            match integrator_type {
                IntegratorType::PathTracing => {
                    sampled_renders.push((IntegratorType::PathTracing, render_settings.clone()))
                }
                t if batched_renders_and_cameras.contains_key(&t) => {
                    // then determine new camera id
                    let mut updated_render_settings = render_settings.clone();
                    updated_render_settings.camera_id = Some(bundled_cameras.len() as u16);
                    // let mut updated_render_settings = RenderSettings {
                    //     camera_id: Some(bundled_cameras.len() as u16),
                    //     ..*render_settings
                    // };

                    // and push to cameras to be used for splatting
                    // bundled_cameras.push(copied_camera);
                    // splatted_renders.push((t, updated_render_settings));
                    batched_renders_and_cameras
                        .get_mut(&t)
                        .unwrap()
                        .push((updated_render_settings, copied_camera))
                }
                _ => {}
            }
        }
        // phase 2, for renders that don't require a splatted render, do them first
        let arc_world = Arc::new(world);
        for (integrator_type, render_settings) in sampled_renders.iter() {
            match integrator_type {
                IntegratorType::PathTracing => {
                    if let Some(Integrator::PathTracing(integrator)) =
                        Integrator::from_settings_and_world(
                            arc_world.clone(),
                            IntegratorType::PathTracing,
                            &bundled_cameras,
                            render_settings,
                        )
                    {
                        // let integrator: Box<dyn SamplerIntegrator> = Box::new(integrator);
                        films.push((
                            render_settings.clone(),
                            NaiveRenderer::render_sampled(
                                integrator,
                                render_settings,
                                &bundled_cameras[render_settings.camera_id.unwrap() as usize],
                            ),
                        ));
                    }
                }
                _ => {}
            }
        }

        // phase 3, do renders where cameras can be combined

        for integrator_type in batched_renders_and_cameras.keys() {
            match integrator_type {
                IntegratorType::BDPT => {
                    let (bundled_settings, bundled_cameras): (Vec<RenderSettings>, Vec<Camera>) =
                        batched_renders_and_cameras
                            .get(integrator_type)
                            .unwrap()
                            .iter()
                            .cloned()
                            .unzip();
                    let integrator = BDPTIntegrator {
                        max_bounces: 10,
                        world: arc_world.clone(),
                        specific_pair: None,
                        cameras: bundled_cameras.clone(),
                    };

                    // let integrator: Box<dyn GenericIntegrator> = Box::new(integrator);
                    films.extend((&bundled_settings).iter().cloned().zip(
                        NaiveRenderer::render_splatted(
                            integrator,
                            bundled_settings.clone(),
                            bundled_cameras,
                        ),
                    ));
                }
                _ => {}
            }
        }

        // phase 4: tonemap and output all films

        for (render_settings, film) in films.iter() {
            let filename = render_settings.filename.as_ref();
            let filename_str = filename.cloned().unwrap_or(String::from("output"));
            let exr_filename = format!("output/{}.exr", filename_str);
            let png_filename = format!("output/{}.png", filename_str);

            let srgb_tonemapper = sRGB::new(&film, 1.0);
            srgb_tonemapper.write_to_files(&film, &exr_filename, &png_filename);
        }
    }
}
