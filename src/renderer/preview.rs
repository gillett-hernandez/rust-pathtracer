use super::{output_film, Film, Renderer};

use crate::parsing::config::{Config, IntegratorKind, RenderSettings, RendererType, Resolution};
// use crate::integrator::*;
use crate::camera::Camera;
use crate::integrator::{
    CameraId, GenericIntegrator, Integrator, IntegratorType, Sample, SamplerIntegrator,
};
use crate::math::{RandomSampler, Sampler, StratifiedSampler, XYZColor};
use crate::parsing::parse_tonemapper;
use crate::profile::Profile;
use crate::rgb_to_u32;
use crate::world::World;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::unbounded;
use pbr::ProgressBar;

use minifb::{Key, Scale, Window, WindowOptions};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

#[derive(Default)]
pub struct PreviewRenderer {}

impl PreviewRenderer {
    pub fn new() -> Self {
        PreviewRenderer {}
    }
}
impl Renderer for PreviewRenderer {
    fn render(&self, mut world: World, cameras: Vec<Camera>, config: &Config) {
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
            let (mut tonemapper, converter) = parse_tonemapper(render_settings.tonemap_settings);

            world.assign_cameras(vec![cameras[render_settings.camera_id].clone()], false);
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
            match Integrator::from_settings_and_world(
                arc_world,
                IntegratorType::from(render_settings.integrator),
                &cameras,
                &render_settings,
            ) {
                None => {}
                Some(Integrator::BDPT(mut integrator)) => {
                    println!("rendering with BDPT integrator");
                    let now = Instant::now();

                    let mut total_camera_samples = 0;
                    let mut total_pixels = 0;
                    let light_film: Arc<Mutex<Film<XYZColor>>> =
                        Arc::new(Mutex::new(Film::new(width, height, XYZColor::BLACK)));

                    let (width, height) = (
                        render_settings.resolution.width,
                        render_settings.resolution.height,
                    );
                    println!("starting render with film resolution {}x{}", width, height);
                    let pixels = width * height;
                    total_pixels += pixels;
                    total_camera_samples += pixels * (render_settings.min_samples as usize);

                    println!("total pixels: {}", total_pixels);
                    println!("minimum total samples: {}", total_camera_samples);
                    let maximum_threads = render_settings.threads.unwrap_or(1);

                    const SHOW_PROGRESS_BAR: bool = true;

                    let mut sampler: Box<dyn Sampler> =
                        Box::new(StratifiedSampler::new(20, 20, 10));
                    let mut preprocess_profile = Profile::default();
                    integrator.preprocess(
                        &mut sampler,
                        &[render_settings.clone()],
                        &mut preprocess_profile,
                    );
                    let mut pb = ProgressBar::new(total_pixels as u64);

                    let pixel_count = Arc::new(AtomicUsize::new(0));
                    let pixel_count_clone = pixel_count.clone();
                    let thread = thread::spawn(move || {
                        let mut local_index = 0;
                        while local_index < total_pixels {
                            let pixels_to_increment =
                                pixel_count_clone.load(Ordering::Relaxed) - local_index;
                            if SHOW_PROGRESS_BAR {
                                pb.add(pixels_to_increment as u64);
                            }
                            local_index += pixels_to_increment;

                            thread::sleep(Duration::from_millis(250));
                        }
                    });

                    let (tx, rx) = unbounded();
                    // let (tx, rx) = bounded(100000);

                    let total_splats = Arc::new(Mutex::new(0usize));
                    let stop_splatting = Arc::new(AtomicBool::new(false));

                    let light_film_ref = Arc::clone(&light_film);
                    let total_splats_ref = Arc::clone(&total_splats);
                    let stop_splatting_ref = Arc::clone(&stop_splatting);
                    let mut light_window = Window::new(
                        "Light Preview",
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
                    light_window.limit_update_rate(Some(std::time::Duration::from_micros(16666)));
                    let light_buffer = Arc::new(Mutex::new(vec![0u32; width * height]));
                    let light_buffer_ref = Arc::clone(&light_buffer);
                    let render_settings_copy = render_settings.clone();

                    // TODO: write abstraction for splatting threads,
                    // by passing in the Arc mutex of the relevant destination film and the relevant channel
                    let splatting_thread = thread::spawn(move || {
                        let film = &mut light_film_ref.lock().unwrap();
                        let mut local_total_splats = total_splats_ref.lock().unwrap();
                        let mut local_stop_splatting = false;
                        let mut remaining_iterations = 10;
                        let (mut tonemapper2, converter) =
                            parse_tonemapper(render_settings_copy.tonemap_settings);

                        loop {
                            // let mut samples: Vec<(Sample, u8)> = rx.try_iter().collect();
                            // samples.par_sort_unstable_by(|(a0, b0), (a1, b1)| {
                            //     // primary sort by camera id
                            //     match b0.cmp(b1) {
                            //         CmpOrdering::Equal => match (a0, a1) {
                            //             // if camera ids match, secondary sort by pixel
                            //             (
                            //                 Sample::LightSample(_, (p0x, p0y)),
                            //                 Sample::LightSample(_, (p1x, p1y)),
                            //             ) => {
                            //                 // let (p0x, p0y) = a0.1;
                            //                 // let (p1x, p1y) = a1.1;
                            //                 if p0y < p1y {
                            //                     CmpOrdering::Less
                            //                 } else if p0y > p1y {
                            //                     CmpOrdering::Greater
                            //                 } else {
                            //                     if p0x < p1x {
                            //                         CmpOrdering::Less
                            //                     } else if p0x > p1x {
                            //                         CmpOrdering::Greater
                            //                     } else {
                            //                         CmpOrdering::Equal
                            //                     }
                            //                 }
                            //             }
                            //             _ => CmpOrdering::Equal,
                            //         },
                            //         t => t,
                            //     }
                            // });
                            for v in rx.try_iter() {
                                let (sample, _film_id): (Sample, CameraId) = v;

                                if let Sample::LightSample(sample, pixel) = sample {
                                    let color = sample;
                                    let (width, height) = (film.width, film.height);
                                    let (x, y) = (
                                        (pixel.0 * width as f32) as usize,
                                        height - (pixel.1 * height as f32) as usize - 1,
                                    );

                                    film.buffer[y * width + x] += color;
                                    (*local_total_splats) += 1usize;
                                }
                            }

                            {
                                tonemapper2.initialize(film, 1.0);
                                light_buffer_ref
                                    .lock()
                                    .unwrap()
                                    .par_iter_mut()
                                    .enumerate()
                                    .for_each(|(pixel_idx, v)| {
                                        let y: usize = pixel_idx / width;
                                        let x: usize = pixel_idx - width * y;
                                        let [r, g, b, _]: [f32; 4] = converter
                                            .transfer_function(
                                                tonemapper2.map(film, (x as usize, y as usize)),
                                                false,
                                            )
                                            .into();
                                        *v = rgb_to_u32(
                                            (255.0 * r) as u8,
                                            (255.0 * g) as u8,
                                            (255.0 * b) as u8,
                                        );
                                    });
                            }
                            if !local_stop_splatting && stop_splatting_ref.load(Ordering::Relaxed) {
                                local_stop_splatting = true;
                            }
                            if local_stop_splatting {
                                remaining_iterations -= 1;
                            }
                            thread::sleep(Duration::from_millis(100));
                            if remaining_iterations <= 0 {
                                break;
                            }
                        }
                    });

                    let tx_arc = Arc::new(Mutex::new(tx));
                    // might need to rate limit based on the speed at which splatting is occurring, but for now don't limit.
                    // Light tracing will use an unbounded amount of memory though.
                    let per_splat_sleep_time = Duration::from_nanos(0);

                    if let IntegratorKind::BDPT {
                        selected_pair: Some((s, t)),
                    } = render_settings.clone().integrator
                    {
                        println!("rendering specific pair {} {}", s, t);
                    }

                    let mut s = 0;
                    loop {
                        if !window.is_open()
                            || window.is_key_down(Key::Escape)
                            || !light_window.is_open()
                            || light_window.is_key_down(Key::Escape)
                        {
                            break;
                        }
                        films[film_idx].buffer.par_iter_mut().enumerate().for_each(
                            |(pixel_index, pixel_ref)| {
                                let mut profile = Profile::default();
                                let tx1 = { tx_arc.lock().unwrap().clone() };
                                let y: usize = pixel_index / render_settings.resolution.width;
                                let x: usize = pixel_index - render_settings.resolution.width * y;

                                let mut temp_color = XYZColor::BLACK;
                                let mut sampler: Box<dyn Sampler> =
                                    Box::new(StratifiedSampler::new(20, 20, 10));
                                // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                                // idea: use SPD::Tabulated to collect all the data for a single pixel as a SPD, then convert that whole thing to XYZ.
                                let mut local_additional_splats: Vec<(Sample, CameraId)> =
                                    Vec::new();
                                // use with capacity to preallocate
                                let sample = sampler.draw_2d();
                                let camera_uv = (
                                    ((x as f32 + sample.x)
                                        / (render_settings.resolution.width as f32))
                                        .clamp(0.0, 1.0 - std::f32::EPSILON),
                                    ((y as f32 + sample.y)
                                        / (render_settings.resolution.height as f32))
                                        .clamp(0.0, 1.0 - std::f32::EPSILON),
                                );
                                temp_color += integrator.color(
                                    &mut sampler,
                                    &render_settings,
                                    (camera_uv, render_settings.camera_id),
                                    s as usize,
                                    &mut local_additional_splats,
                                    &mut profile,
                                );

                                debug_assert!(
                                    temp_color.0.is_finite().all(),
                                    "integrator returned {:?}",
                                    temp_color
                                );

                                *pixel_ref += temp_color;
                                pixel_count.fetch_add(1, Ordering::Relaxed);
                                if per_splat_sleep_time.as_nanos() > 0 {
                                    thread::sleep(
                                        per_splat_sleep_time * local_additional_splats.len() as u32,
                                    );
                                }
                                for splat in local_additional_splats {
                                    tx1.send(splat).unwrap();
                                }
                            },
                        );
                        tonemapper.initialize(&films[film_idx], 1.0 / (s as f32 + 1.0));
                        buffer
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(pixel_idx, v)| {
                                let y: usize = pixel_idx / width;
                                let x: usize = pixel_idx - width * y;
                                let [r, g, b, _]: [f32; 4] = converter
                                    .transfer_function(
                                        tonemapper.map(&films[film_idx], (x as usize, y as usize)),
                                        false,
                                    )
                                    .into();
                                *v = rgb_to_u32(
                                    (255.0 * r) as u8,
                                    (255.0 * g) as u8,
                                    (255.0 * b) as u8,
                                );
                            });
                        window.update_with_buffer(&buffer, width, height).unwrap();

                        light_window
                            .update_with_buffer(&light_buffer.lock().unwrap(), width, height)
                            .unwrap();
                        s += 1;
                    }

                    if let Err(panic) = thread.join() {
                        println!(
                            "progress bar incrementing thread threw an error {:?}",
                            panic
                        );
                    }

                    let now2 = Instant::now();
                    stop_splatting.store(true, Ordering::Relaxed);

                    if let Err(panic) = splatting_thread.join() {
                        println!("panic occurred within thread: {:?}", panic);
                    }
                    let elapsed2 = (now2.elapsed().as_millis() as f32) / 1000.0;
                    println!(
                        "found {} splats, and took {}s to finish splatting them to film",
                        total_splats.lock().unwrap(),
                        elapsed2
                    );

                    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;

                    println!("took {}s", elapsed);
                    preprocess_profile.pretty_print(elapsed, maximum_threads as usize);
                    output_film(&render_settings, &films[film_idx], 1.0 / (s as f32 + 1.0));
                    output_film(
                        &RenderSettings {
                            filename: Some(format!(
                                "{}{}",
                                render_settings.filename.unwrap(),
                                "_lightfilm"
                            )),
                            ..render_settings
                        },
                        &light_film.lock().unwrap(),
                        1.0,
                    );
                }
                Some(Integrator::PathTracing(integrator)) => {
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

                    let mut s = 0;
                    loop {
                        if !window.is_open() || window.is_key_down(Key::Escape) {
                            break;
                        }
                        let now = Instant::now();
                        let _stats: Profile = films[film_idx]
                            .buffer
                            .par_iter_mut()
                            .enumerate()
                            .map(|(pixel_index, pixel_ref)| {
                                let mut profile = Profile::default();
                                // let clone = pixel_count.clone();
                                let y: usize = pixel_index / width;
                                let x: usize = pixel_index - width * y;
                                // gen ray for pixel x, y

                                let mut temp_color = XYZColor::BLACK;
                                // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
                                let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());

                                let sample = sampler.draw_2d();

                                let camera_uv = (
                                    (x as f32 + sample.x)
                                        / (render_settings.resolution.width as f32),
                                    (y as f32 + sample.y)
                                        / (render_settings.resolution.height as f32),
                                );
                                temp_color += integrator.color(
                                    &mut sampler,
                                    (camera_uv, 0),
                                    s as usize,
                                    &mut profile,
                                );

                                debug_assert!(
                                    temp_color.0.is_finite().all(),
                                    "{:?} resulted in {:?}",
                                    camera_uv,
                                    temp_color
                                );

                                pixel_count.fetch_add(1, Ordering::Relaxed);

                                *pixel_ref += temp_color;

                                profile
                            })
                            .reduce(Profile::default, |a, b| a.combine(b));
                        // stats.pretty_print(elapsed, render_settings.threads.unwrap() as usize);
                        println!();
                        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;
                        println!("fps {}", 1.0 / elapsed);
                        tonemapper.initialize(&films[film_idx], 1.0 / (s as f32 + 1.0));
                        buffer
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(pixel_idx, v)| {
                                let y: usize = pixel_idx / width;
                                let x: usize = pixel_idx - width * y;
                                let [r, g, b, _]: [f32; 4] = converter
                                    .transfer_function(
                                        tonemapper.map(&films[film_idx], (x as usize, y as usize)),
                                        false,
                                    )
                                    .into();
                                *v = rgb_to_u32(
                                    (256.0 * r) as u8,
                                    (256.0 * g) as u8,
                                    (256.0 * b) as u8,
                                );
                            });
                        window.update_with_buffer(&buffer, width, height).unwrap();
                        s += 1;
                    }
                    if let Err(panic) = thread.join() {
                        println!(
                            "progress bar incrememnting thread threw an error {:?}",
                            panic
                        );
                    }
                    output_film(&render_settings, &films[film_idx], 1.0 / (s as f32 + 1.0));
                }
                Some(_) => {}
            }
        }
    }
}
