mod film;

pub use film::Film;

use crate::camera::Camera;
use crate::config::Config;
use crate::config::RenderSettings;
use crate::integrator::*;
use crate::math::*;
use crate::world::World;

use std::io::Write;
use std::sync::Arc;

use rayon::prelude::*;

fn construct_integrator(
    settings: &RenderSettings,
    world: Arc<World>,
) -> Box<dyn SamplerIntegrator> {
    let max_bounces = settings.max_bounces.unwrap_or(1);
    let russian_roulette = settings.russian_roulette.unwrap_or(true);
    let light_samples = settings.light_samples.unwrap_or(4);
    let only_direct = settings.only_direct.unwrap_or(false);

    match settings
        .integrator
        .as_ref()
        .unwrap_or(&"PT".to_string())
        .as_ref()
    {
        "PT" => {
            println!(
                "constructing path tracing integrator, max bounces: {},\nrussian_roulette: {}, light_samples: {}",
                max_bounces, russian_roulette, light_samples
            );
            Box::new(PathTracingIntegrator {
                max_bounces,
                world,
                russian_roulette,
                light_samples,
                only_direct,
            })
        }
        "LT" => {
            println!("constructing light tracing integrator");
            Box::new(LightTracingIntegrator {
                max_bounces,
                world,
                russian_roulette,
            })
        }
        "BDPT" => {
            println!(
                "constructing BDPT integrator with selected pair {:?}",
                settings.selected_pair
            );
            Box::new(BDPTIntegrator {
                max_bounces,
                world,
                specific_pair: settings.selected_pair,
            })
        }
        _ => Box::new(PathTracingIntegrator {
            max_bounces,
            world,
            russian_roulette,
            light_samples,
            only_direct,
        }),
    }
}
pub struct NaiveRenderer {}

impl NaiveRenderer {
    pub fn new() -> NaiveRenderer {
        NaiveRenderer {}
    }
}

pub trait Renderer {
    fn render(&self, cameras: &Vec<Box<dyn Camera>>, settings: &Config, film: &mut Film<XYZColor>);
}

impl Renderer for NaiveRenderer {
    fn render(&self, camera: &Vec<Box<dyn Camera>>, config: &Config) -> Vec<Film<XYZColor>> {
        // get settings for each film
        for (_render_id, render_settings) in config.render_settings.iter().enumerate() {
            let camera_id = render_settings.camera_id.unwrap_or(0) as usize;

            let (width, height) = (
                render_settings.resolution.width,
                render_settings.resolution.height,
            );
            println!("starting render with film resolution {}x{}", width, height);
            let min_camera_rays = width * height * render_settings.min_samples as usize;
            println!("minimum total samples: {}", min_camera_rays);

            let aspect_ratio = width as f32 / height as f32;

            let now = Instant::now();

            &cameras[camera_id].modify_aspect_ratio(aspect_ratio);

            let film = renderer.render(&cameras[camera_id], &render_settings, &world);

            let total_camera_rays = film.total_pixels()
                * (render_settings
                    .max_samples
                    .unwrap_or(render_settings.min_samples) as usize);

            let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;

            // do stuff with film here

            let filename = render_settings.filename.as_ref();
            let filename_str = filename.cloned().unwrap_or(String::from("output"));
            let exr_filename = format!("output/{}.exr", filename_str);
            let png_filename = format!("output/{}.png", filename_str);

            let srgb_tonemapper = tonemap::sRGB::new(&film, 10.0);
            srgb_tonemapper.write_to_files(&film, &exr_filename, &png_filename);
            println!(
                "\ntook {}s at {} rays per second and {} rays per second per thread",
                elapsed,
                (total_camera_rays as f32) / elapsed,
                (total_camera_rays as f32) / elapsed / (render_settings.threads.unwrap() as f32)
            );
        }
        // for y in 0..film.height {
        //     for x in 0..film.width {
        let width = film.width;
        let height = film.height;

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
    }
}
