mod film;

pub use film::Film;

use crate::camera::Camera;
use crate::config::Config;
use crate::config::RenderSettings;
use crate::integrator::*;
use crate::math::*;
use crate::tonemap::sRGB;
use crate::world::World;

use std::io::Write;
// use std::sync::Arc;
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

pub struct NaiveRenderer {
    world: World,
}

impl NaiveRenderer {
    pub fn new(world: World) -> NaiveRenderer {
        NaiveRenderer { world }
    }

    pub fn render_sampled(
        integrator: Integrator,
        settings: &RenderSettings,
        camera: &Box<dyn Camera>,
    ) -> Film<XYZColor> {
        let (width, height) = (settings.resolution.width, settings.resolution.height);
        let film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);
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
    pub fn render_splatted(
        integrator: Integrator,
        renders: Vec<RenderSettings>,
    ) -> Vec<Film<XYZColor>> {
        Vec::new()
    }
}

pub trait Renderer {
    fn render(&self, cameras: Vec<Box<dyn Camera>>, config: &Config);
}

impl Renderer for NaiveRenderer {
    fn render(&self, mut cameras: Vec<Box<dyn Camera>>, config: &Config) {
        // bin the render settings into bins corresponding to what integrator they need.

        let mut bundled_cameras: Vec<Box<dyn Camera>> = Vec::new();
        let mut splatted_renders: Vec<RenderSettings> = Vec::new();
        let mut films: Vec<(RenderSettings, Film<XYZColor>)> = Vec::new();
        let mut sampled_renders: Vec<RenderSettings> = Vec::new();

        // phase 1, gather and sort what renders need to be done
        for (_render_id, mut render_settings) in config.render_settings.iter().enumerate() {
            let camera_id = render_settings.camera_id.unwrap_or(0) as usize;

            let (width, height) = (
                render_settings.resolution.width,
                render_settings.resolution.height,
            );
            let aspect_ratio = width as f32 / height as f32;

            // copy camera and modify its aspect ratio (so that uv splatting works correctly)
            let mut copied_camera = cameras[camera_id].copy();
            copied_camera.modify_aspect_ratio(aspect_ratio);

            match IntegratorType::from(render_settings.integrator.unwrap_or("PT".into()).into()) {
                IntegratorType::PathTracing => sampled_renders.push(render_settings.clone()),
                _ => {
                    // then determine new camera id
                    render_settings.camera_id = Some(bundled_cameras.len() as u16);

                    // and push to cameras to be used for splatting
                    bundled_cameras.push(copied_camera);
                    splatted_renders.push(render_settings.clone());
                }
            }
        }
        // phase 2, for renders that don't require a splatted render, do them first
        for render in sampled_renders.iter() {
            films.push((render, self.render_sampled(render)));
        }

        // phase 3, do renders where cameras can be combined

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
