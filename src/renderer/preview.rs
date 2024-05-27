// only needed because the actual code that uses these imports is gated
// behind a feature flag which the compiler does not seem to detect even though
#![allow(unused_imports)]
use super::prelude::*;

use crate::parsing::config::{Config, RenderSettings, RendererType, Resolution};
// use crate::integrator::*;

use crate::integrator::{GenericIntegrator, Integrator, IntegratorType, Sample, SamplerIntegrator};

use crate::parsing::parse_tonemap_settings;
use crate::profile::Profile;
use crate::world::{EnvironmentMap, ImportanceMap, World};

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::unbounded;
use math::spectral::BOUNDED_VISIBLE_RANGE;
use pbr::ProgressBar;

#[cfg(feature = "preview")]
use minifb::{Key, Scale, Window, WindowOptions};
use rayon::iter::ParallelIterator;

#[cfg(feature = "preview")]
#[derive(Default)]
pub struct PreviewRenderer {}

#[cfg(feature = "preview")]
impl PreviewRenderer {
    pub fn new() -> Self {
        PreviewRenderer {}
    }
}

#[cfg(feature = "preview")]
impl Renderer for PreviewRenderer {
    fn render(&self, mut world: World, config: &Config) {
        if let RendererType::Preview {
            selected_preview_film_id,
        } = config.renderer
        {
            let mut films: Vec<Vec2D<XYZColor>> = Vec::new();
            for render_settings in config.render_settings.iter() {
                let Resolution { width, height } = render_settings.resolution;
                films.push(Vec2D::new(width, height, XYZColor::BLACK));
            }
            println!("num cameras {}", world.cameras.len());

            let film_idx = selected_preview_film_id;
            let original_render_settings = config.render_settings[film_idx].clone();
            let mut render_settings = config.render_settings[film_idx].clone();
            render_settings.tonemap_settings = render_settings.tonemap_settings.silenced();
            let mut tonemapper = parse_tonemap_settings(render_settings.tonemap_settings);

            let env_sampling_probability = world.get_env_sampling_probability();
            if let EnvironmentMap::HDR {
                texture,
                importance_map,
                strength,
                ..
            } = &mut world.environment
            {
                if *strength > 0.0
                    && env_sampling_probability > 0.0
                    && !matches!(importance_map, ImportanceMap::Baked { .. })
                {
                    importance_map.bake_in_place(
                        texture,
                        render_settings
                            .wavelength_bounds
                            .map(|e| e.into())
                            .unwrap_or(BOUNDED_VISIBLE_RANGE),
                    );
                }
            }
            let arc_world = Arc::new(world.clone());

            let Resolution { width, height } = render_settings.resolution;

            match Integrator::from_settings_and_world(
                arc_world,
                IntegratorType::from(render_settings.integrator),
                &world.cameras[..],
                &render_settings,
            ) {
                None => {}
                /* Some(Integrator::BDPT(mut integrator)) => {
                    println!("rendering with BDPT integrator");
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
                    window.set_target_fps(60);
                    let mut buffer = vec![0u32; width * height];
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
                    light_window.set_target_fps(60);
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
                            // let srgb_tonemapper = sRGB::new(
                            //     &film,
                            //     render_settings_copy.exposure.unwrap_or(1.0),
                            //     false,
                            // );
                            {
                                update_window_buffer(
                                    &mut light_buffer_ref.lock().unwrap(),
                                    &film,
                                    tonemapper2.as_mut(),
                                    converter,
                                    1.0,
                                );
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
                                // idea: use Curve::Tabulated to collect all the data for a single pixel as a Curve, then convert that whole thing to XYZ.
                                let mut local_additional_splats: Vec<(Sample, CameraId)> =
                                    Vec::new();
                                // use with capacity to preallocate
                                let sample = sampler.draw_2d();
                                let camera_uv = (
                                    ((x as f32 + sample.x)
                                        / (render_settings.resolution.width as f32))
                                        .clamp(0.0, 1.0 - f32::EPSILON),
                                    ((y as f32 + sample.y)
                                        / (render_settings.resolution.height as f32))
                                        .clamp(0.0, 1.0 - f32::EPSILON),
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
                        update_window_buffer(
                            &mut buffer,
                            &films[film_idx],
                            tonemapper.as_mut(),
                            converter,
                            1.0 / (s as f32 + 1.0),
                        );
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
                } */
                Some(Integrator::PathTracing(integrator)) => {
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
                    window.set_target_fps(60);
                    let mut buffer = vec![0u32; width * height];

                    let max_samples = render_settings
                        .max_samples
                        .unwrap_or(render_settings.min_samples);
                    let min_camera_rays = width * height * render_settings.min_samples as usize;
                    println!("minimum total samples: {}", min_camera_rays);

                    let mut pb = ProgressBar::new((max_samples as usize * width * height) as u64);

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
                        if !window.is_open()
                            || window.is_key_down(Key::Escape)
                            || (s > max_samples && max_samples > 0)
                        {
                            break;
                        }

                        if s > max_samples && max_samples > 0 {
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
                                let x: usize = pixel_index % width;
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

                                *pixel_ref += temp_color;

                                profile
                            })
                            .reduce(Profile::default, |a, b| a.combine(b));

                        pixel_count.fetch_add(width * height, Ordering::Relaxed);
                        // stats.pretty_print(elapsed, render_settings.threads.unwrap() as usize);
                        println!();
                        let elapsed_calc = (now.elapsed().as_millis() as f32) / 1000.0;

                        update_window_buffer(
                            &mut buffer,
                            &films[film_idx],
                            tonemapper.as_mut(),
                            1.0 / (s as f32 + 1.0),
                        );
                        window.update_with_buffer(&buffer, width, height).unwrap();

                        let elapsed_total = (now.elapsed().as_millis() as f32) / 1000.0;
                        let elapsed_tonemap_update = elapsed_total - elapsed_calc;
                        println!(
                            "fps {}, num samples {}. % time taken by tonemap_and_update: {}",
                            1.0 / elapsed_total,
                            s,
                            100.0 * elapsed_tonemap_update / elapsed_total
                        );
                        s += 1;
                    }
                    if let Err(panic) = thread.join() {
                        println!(
                            "progress bar incrememnting thread threw an error {:?}",
                            panic
                        );
                    }
                    output_film(&render_settings, &films[film_idx], 1.0 / (s as f32 + 1.0));
                    println!("total samples: {}", s);
                }
                Some(Integrator::LightTracing(mut integrator)) => {
                    println!("rendering with LT integrator");
                    let now = Instant::now();

                    let mut total_camera_samples = 0;
                    let mut total_pixels = 0;
                    let light_film: Arc<Mutex<Vec2D<XYZColor>>> =
                        Arc::new(Mutex::new(Vec2D::new(width, height, XYZColor::BLACK)));

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

                    let mut sampler: Box<dyn Sampler> =
                        Box::new(StratifiedSampler::new(20, 20, 10));
                    let mut preprocess_profile = Profile::default();
                    integrator.preprocess(
                        &mut sampler,
                        &[render_settings.clone()],
                        &mut preprocess_profile,
                    );

                    let (tx, rx) = unbounded();

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
                    light_window.set_target_fps(60);
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
                        let mut tonemapper2 =
                            parse_tonemap_settings(render_settings_copy.tonemap_settings);

                        let mut last_instant = Instant::now();
                        let first_instant = last_instant.clone();
                        let mut last_total_splats = 0;
                        loop {
                            for v in rx.try_iter() {
                                let (sample, _film_id): (Sample, CameraId) = v;

                                if let Sample::LightSample(sample, pixel) = sample {
                                    let color = sample;
                                    let (width, height) = (film.width, film.height);
                                    let (x, y) = (
                                        (pixel.0 * width as f32) as usize,
                                        // vertical inversion
                                        // height - (pixel.1 * height as f32) as usize - 1,
                                        ((1.0 - pixel.1) * height as f32) as usize,
                                    );

                                    film.buffer[y * width + x] += color;
                                    (*local_total_splats) += 1usize;
                                }
                            }

                            {
                                let owned = local_total_splats.to_owned();
                                #[cfg(feature = "preview")]
                                update_window_buffer(
                                    &mut light_buffer_ref.lock().unwrap(),
                                    &film,
                                    tonemapper2.as_mut(),
                                    1.0 / ((owned as f32).sqrt() + 1.0),
                                );
                                println!(
                                    "total splats {}, {} per second, avg: {}",
                                    owned,
                                    1000.0 * (owned - last_total_splats) as f32
                                        / last_instant.elapsed().as_millis() as f32,
                                    1000.0 * owned as f32
                                        / first_instant.elapsed().as_millis() as f32
                                );

                                last_instant = Instant::now();
                                last_total_splats = owned;
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

                        // total_splats_ref
                    });

                    let tx_arc = Arc::new(Mutex::new(tx));
                    // might need to rate limit based on the speed at which splatting is occurring, but for now don't limit.
                    // Light tracing will use an unbounded amount of memory though.
                    let per_splat_sleep_time = Duration::from_nanos(0);

                    let mut s = 0;
                    let mut profile = Profile::default();
                    loop {
                        if !light_window.is_open() || light_window.is_key_down(Key::Escape) {
                            break;
                        }
                        let local_profile = films[film_idx]
                            .buffer
                            .par_iter_mut()
                            .enumerate()
                            .map(|(pixel_index, pixel_ref)| {
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
                                        .clamp(0.0, 1.0 - f32::EPSILON),
                                    ((y as f32 + sample.y)
                                        / (render_settings.resolution.height as f32))
                                        .clamp(0.0, 1.0 - f32::EPSILON),
                                );
                                temp_color += integrator.color(
                                    &mut sampler,
                                    &render_settings,
                                    (
                                        camera_uv,
                                        config.camera_names_to_index[&render_settings.camera_id],
                                    ),
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
                                if per_splat_sleep_time.as_nanos() > 0 {
                                    thread::sleep(
                                        per_splat_sleep_time * local_additional_splats.len() as u32,
                                    );
                                }
                                for splat in local_additional_splats {
                                    tx1.send(splat).unwrap();
                                }
                                profile
                            })
                            .reduce(|| Profile::default(), |a, b| a.combine(b));

                        light_window
                            .update_with_buffer(&light_buffer.lock().unwrap(), width, height)
                            .unwrap();
                        s += 1;
                        profile = profile.combine(local_profile);
                    }

                    let now2 = Instant::now();
                    stop_splatting.store(true, Ordering::Relaxed);

                    if let Err(panic) = splatting_thread.join() {
                        println!("panic occurred within thread: {:?}", panic);
                    }
                    let elapsed2 = (now2.elapsed().as_millis() as f32) / 1000.0;
                    println!(
                        "found {} total splats, and took {}s to finish splatting them to film after giving the stop command",
                        total_splats.lock().unwrap(),
                        elapsed2
                    );

                    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;

                    println!("took {}s\npreprocess profile:", elapsed);
                    preprocess_profile.pretty_print(elapsed, maximum_threads as usize);
                    println!("full profile");
                    profile.pretty_print(elapsed, maximum_threads as usize);

                    // output_film(
                    //     &original_render_settings,
                    //     &films[film_idx],
                    //     1.0 / (s as f32 + 1.0),
                    // );
                    output_film(
                        &RenderSettings {
                            filename: Some(format!(
                                "{}{}",
                                render_settings.filename.unwrap(),
                                "_lightfilm"
                            )),
                            ..original_render_settings
                        },
                        &light_film.lock().unwrap(),
                        1.0 / (s as f32 + 1.0),
                    );
                }
            }
        }
    }
    fn supported_integrators(&self) -> &[IntegratorKind] {
        &[
            IntegratorKind::PT {
                light_samples: 0,
                medium_aware: false,
            },
            IntegratorKind::LT { camera_samples: 0 },
        ]
    }
}
