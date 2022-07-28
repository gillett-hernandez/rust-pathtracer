use super::{calculate_widest_wavelength_bounds, output_film, Film, Renderer};

use crate::camera::{Camera, CameraId};

use crate::config::{Config, IntegratorKind, RenderSettings};
use crate::integrator::{
    BDPTIntegrator, GenericIntegrator, Integrator, IntegratorType, LightTracingIntegrator, Sample,
    SamplerIntegrator,
};
use crate::math::{RandomSampler, Sampler, StratifiedSampler, XYZColor};
use crate::profile::Profile;
use crate::world::{EnvironmentMap, World};

use math::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;
use math::Bounds1D;

use std::collections::HashMap;
// use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::unbounded;
// use crossbeam::channel::{bounded};
use pbr::ProgressBar;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

pub struct NaiveRenderer {}

impl NaiveRenderer {
    pub fn new() -> NaiveRenderer {
        NaiveRenderer {}
    }

    pub fn render_sampled<I: SamplerIntegrator>(
        mut integrator: I,
        settings: &RenderSettings,
        _camera: &Camera,
    ) -> Film<XYZColor> {
        let (width, height) = (settings.resolution.width, settings.resolution.height);
        println!("starting render with film resolution {}x{}", width, height);
        let min_camera_rays = width * height * settings.min_samples as usize;
        println!(
            "minimum samples per pixel: {}",
            settings.min_samples as usize
        );
        println!("minimum total samples: {}", min_camera_rays);

        let now = Instant::now();

        let mut film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);

        let mut pb = ProgressBar::new((width * height) as u64);

        let mut presampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
        let mut preprofile = Profile::default();
        integrator.preprocess(&mut presampler, &vec![settings.clone()], &mut preprofile);

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
        let stats: Profile = film
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

                for s in 0..settings.min_samples {
                    let sample = sampler.draw_2d();

                    let camera_uv = (
                        (x as f32 + sample.x) / (settings.resolution.width as f32),
                        (y as f32 + sample.y) / (settings.resolution.height as f32),
                    );
                    temp_color +=
                        integrator.color(&mut sampler, (camera_uv, 0), s as usize, &mut profile);

                    debug_assert!(
                        temp_color.0.is_finite().all(),
                        "{:?} resulted in {:?}",
                        camera_uv,
                        temp_color
                    );
                }

                clone2.fetch_add(1, Ordering::Relaxed);

                *pixel_ref = temp_color / (settings.min_samples as f32);

                profile
            })
            .reduce(|| Profile::default(), |a, b| a.combine(b));

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
    pub fn render_splatted<I: GenericIntegrator>(
        mut integrator: I,
        renders: Vec<RenderSettings>,
        _cameras: Vec<Camera>,
    ) -> Vec<(RenderSettings, Film<XYZColor>)> {
        let now = Instant::now();

        let mut total_camera_samples = 0;
        let mut total_pixels = 0;
        let mut films: Vec<(RenderSettings, Film<XYZColor>)> = Vec::new();
        let light_films: Arc<Mutex<Vec<Film<XYZColor>>>> = Arc::new(Mutex::new(Vec::new()));
        for settings in renders.iter() {
            let (width, height) = (settings.resolution.width, settings.resolution.height);
            println!("starting render with film resolution {}x{}", width, height);
            let pixels = width * height;
            total_pixels += pixels;
            total_camera_samples += pixels * (settings.min_samples as usize);
            let image_film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);
            let light_film: Film<XYZColor> = Film::new(width, height, XYZColor::BLACK);
            films.push((settings.clone(), image_film));
            light_films.lock().unwrap().push(light_film);
        }
        println!("total pixels: {}", total_pixels);
        println!("minimum total samples: {}", total_camera_samples);
        let maximum_threads = renders
            .iter()
            .max_by_key(|s| s.threads)
            .unwrap()
            .threads
            .unwrap();

        const SHOW_PROGRESS_BAR: bool = true;

        let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
        let mut preprocess_profile = Profile::default();
        integrator.preprocess(&mut sampler, &renders, &mut preprocess_profile);
        let mut pb = ProgressBar::new(total_pixels as u64);

        let pixel_count = Arc::new(AtomicUsize::new(0));
        let clone1 = pixel_count.clone();
        let thread = thread::spawn(move || {
            let mut local_index = 0;
            while local_index < total_pixels {
                let pixels_to_increment = clone1.load(Ordering::Relaxed) - local_index;
                if SHOW_PROGRESS_BAR {
                    pb.add(pixels_to_increment as u64);
                }
                local_index += pixels_to_increment;

                thread::sleep(Duration::from_millis(250));
            }
        });

        let clone2 = pixel_count.clone();

        let (tx, rx) = unbounded();
        // let (tx, rx) = bounded(100000);

        let total_splats = Arc::new(Mutex::new(0usize));
        let stop_splatting = Arc::new(AtomicBool::new(false));

        let light_films_ref = Arc::clone(&light_films);
        let total_splats_ref = Arc::clone(&total_splats);
        let stop_splatting_ref = Arc::clone(&stop_splatting);
        let splatting_thread = thread::spawn(move || {
            let films = &mut light_films_ref.lock().unwrap();
            let mut local_total_splats = total_splats_ref.lock().unwrap();
            let mut local_stop_splatting = false;
            let mut remaining_iterations = 10;
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
                    let (sample, film_id): (Sample, CameraId) = v;
                    match sample {
                        Sample::LightSample(sw, pixel) => {
                            let film = &mut films[film_id as usize];
                            let color = sw;
                            let (x, y) = (
                                (pixel.0 * film.width as f32) as usize,
                                film.height - (pixel.1 * film.height as f32) as usize - 1,
                            );

                            film.buffer[y * film.width + x] += color;
                            (*local_total_splats) += 1usize;
                        }
                        _ => {}
                    }
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

        let stats: Vec<Profile> = films
            .par_iter_mut()
            .enumerate()
            .map(
                |(camera_id, (settings, film)): (usize, &mut (RenderSettings, Film<XYZColor>))| {
                    if let IntegratorKind::BDPT {
                        selected_pair: Some((s, t)),
                    } = settings.integrator
                    {
                        println!("rendering specific pair {} {}", s, t);
                    }

                    let profile: Profile = film
                        .buffer
                        .par_iter_mut()
                        .enumerate()
                        .map(|(pixel_index, pixel_ref)| {
                            let mut profile = Profile::default();
                            let tx1 = { tx_arc.lock().unwrap().clone() };
                            let y: usize = pixel_index / settings.resolution.width;
                            let x: usize = pixel_index - settings.resolution.width * y;

                            let mut temp_color = XYZColor::BLACK;
                            let mut sampler: Box<dyn Sampler> =
                                Box::new(StratifiedSampler::new(20, 20, 10));
                            // let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                            // idea: use SPD::Tabulated to collect all the data for a single pixel as a SPD, then convert that whole thing to XYZ.
                            let mut local_additional_splats: Vec<(Sample, CameraId)> = Vec::new();
                            // use with capacity to preallocate
                            for s in 0..settings.min_samples {
                                let sample = sampler.draw_2d();
                                let camera_uv = (
                                    ((x as f32 + sample.x) / (settings.resolution.width as f32))
                                        .clamp(0.0, 1.0 - std::f32::EPSILON),
                                    ((y as f32 + sample.y) / (settings.resolution.height as f32))
                                        .clamp(0.0, 1.0 - std::f32::EPSILON),
                                );
                                temp_color += integrator.color(
                                    &mut sampler,
                                    settings,
                                    (camera_uv, camera_id as CameraId),
                                    s as usize,
                                    &mut local_additional_splats,
                                    &mut profile,
                                );

                                debug_assert!(
                                    temp_color.0.is_finite().all(),
                                    "integrator returned {:?}",
                                    temp_color
                                );
                            }

                            *pixel_ref = temp_color / (settings.min_samples as f32);
                            clone2.fetch_add(1, Ordering::Relaxed);
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
                    profile
                },
            )
            .collect();

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
        for profile in stats {
            profile.pretty_print(elapsed, maximum_threads as usize);
        }

        // TODO: do correct lightfilm + imagefilm combination, instead of outputting both

        let mut i = 0;
        for light_film in light_films.lock().unwrap().iter() {
            let mut render_settings = films[i].0.clone();
            let mut image_film = films[i].1.clone();
            let new_filename = format!(
                "{}{}",
                render_settings
                    .filename
                    .expect("render didn't have filename, wtf"),
                "_combined"
            );
            println!("new filename is {}", new_filename);
            render_settings.filename = Some(new_filename);

            image_film
                .buffer
                .par_iter_mut()
                .enumerate()
                .for_each(|(pixel_index, pixel_ref)| {
                    let y: usize = pixel_index / render_settings.resolution.width;
                    let x: usize = pixel_index - render_settings.resolution.width * y;
                    let light_color = light_film.at(x, y);
                    *pixel_ref = *pixel_ref + light_color / (render_settings.min_samples as f32);
                });

            films.push((render_settings, image_film));
            println!(
                "added combination film to films vec, films vec length is now {}",
                films.len()
            );
            i += 1;
        }

        // let mut i = 0;
        for (i, light_film) in light_films.lock().unwrap().iter().enumerate() {
            let mut render_settings = films[i].0.clone();
            let new_filename = format!(
                "{}{}",
                render_settings
                    .filename
                    .expect("render didn't have filename, wtf"),
                "_lightfilm"
            );
            println!("new filename is {}", new_filename);
            render_settings.filename = Some(new_filename);
            films.push((render_settings, light_film.clone()));
            println!(
                "added light film to films vec, films vec length is now {}",
                films.len()
            );
        }

        films
    }
}

impl Renderer for NaiveRenderer {
    fn render(&self, mut world: World, cameras: Vec<Camera>, config: &Config) {
        // bin the render settings into bins corresponding to what integrator they need.

        let mut bundled_cameras: Vec<Camera> = Vec::new();
        // let mut films: Vec<(RenderSettings, Film<XYZColor>)> = Vec::new();
        let mut sampled_renders: Vec<(IntegratorType, RenderSettings)> = Vec::new();
        let mut splatting_renders_and_cameras: HashMap<
            IntegratorType,
            Vec<(RenderSettings, Camera)>,
        > = HashMap::new();
        splatting_renders_and_cameras.insert(IntegratorType::BDPT, Vec::new());
        splatting_renders_and_cameras.insert(IntegratorType::LightTracing, Vec::new());

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
                // IntegratorType::SPPM => {
                //     let mut updated_render_settings = render_settings.clone();
                //     updated_render_settings.camera_id = camera_id;
                //     bundled_cameras.push(copied_camera);
                //     sampled_renders.push((IntegratorType::SPPM, updated_render_settings));
                // }
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
                IntegratorType::PathTracing => {
                    world.assign_cameras(vec![cameras[render_settings.camera_id].clone()], false);

                    if let EnvironmentMap::HDR {
                        texture,
                        importance_map,
                        ..
                    } = &mut world.environment
                    {
                        let wavelength_bounds = render_settings
                            .wavelength_bounds
                            .map(|e| Bounds1D::new(e.0, e.1))
                            .unwrap_or(math::spectral::BOUNDED_VISIBLE_RANGE);
                        importance_map.bake_in_place(texture, wavelength_bounds);
                    }
                    let arc_world = Arc::new(world.clone());
                    match Integrator::from_settings_and_world(
                        arc_world.clone(),
                        IntegratorType::PathTracing,
                        &bundled_cameras,
                        render_settings,
                    ) {
                        Some(Integrator::PathTracing(integrator)) => {
                            println!("rendering with PathTracing integrator");
                            let (render_settings, film) = (
                                render_settings.clone(),
                                NaiveRenderer::render_sampled(
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
                // IntegratorType::SPPM => {
                //     world.assign_cameras(vec![cameras[render_settings.camera_id].clone()], false);
                //     let arc_world = Arc::new(world.clone());
                //     match Integrator::from_settings_and_world(
                //         arc_world.clone(),
                //         IntegratorType::SPPM,
                //         &bundled_cameras,
                //         render_settings,
                //     ) {
                //         Some(Integrator::SPPM(integrator)) => {
                //             println!("rendering with sppm integrator");
                //             let (render_settings, film) = (
                //                 render_settings.clone(),
                //                 NaiveRenderer::render_sampled(
                //                     integrator,
                //                     render_settings,
                //                     &cameras[render_settings.camera_id],
                //                 ),
                //             );
                //             output_film(&render_settings, &film);
                //         }
                //         _ => {}
                //     }
                // }
                _ => {}
            }
        }

        // phase 3, do renders where cameras can be combined, and output results as soon as they're finished

        for integrator_type in splatting_renders_and_cameras.keys() {
            if let Some(l) = splatting_renders_and_cameras.get(integrator_type) {
                if l.is_empty() {
                    continue;
                }
            }
            match integrator_type {
                IntegratorType::BDPT => {
                    let (bundled_settings, bundled_cameras): (Vec<RenderSettings>, Vec<Camera>) =
                        splatting_renders_and_cameras
                            .get(integrator_type)
                            .unwrap()
                            .iter()
                            .cloned()
                            .unzip();
                    let mut max_bounces = 0;

                    for settings in bundled_settings.iter() {
                        max_bounces = max_bounces.max(settings.max_bounces.unwrap_or(2));
                    }
                    let wavelength_bounds =
                        calculate_widest_wavelength_bounds(&bundled_settings, VISIBLE_RANGE);
                    world.assign_cameras(bundled_cameras.clone(), true);
                    if let EnvironmentMap::HDR {
                        texture,
                        importance_map,
                        ..
                    } = &mut world.environment
                    {
                        importance_map.bake_in_place(texture, wavelength_bounds);
                    }
                    let arc_world = Arc::new(world.clone());
                    let integrator = BDPTIntegrator {
                        max_bounces,
                        world: arc_world.clone(),
                        wavelength_bounds,
                    };

                    println!("rendering with BDPT integrator");
                    let render_splatted_result = NaiveRenderer::render_splatted(
                        integrator,
                        bundled_settings.clone(),
                        bundled_cameras.clone(),
                    );
                    assert!(render_splatted_result.len() > 0);
                    // films.extend(
                    //     (&bundled_settings)
                    //         .iter()
                    //         .cloned()
                    //         .zip(render_splatted_result),
                    // );
                    for (mut render_settings, film) in render_splatted_result {
                        // if selected pair, add the pair numbers to the filename automatically
                        if let IntegratorKind::BDPT {
                            selected_pair: Some((s, t)),
                        } = render_settings.integrator
                        {
                            let new_filename = format!(
                                "{}{}_{}",
                                render_settings
                                    .filename
                                    .expect("render didn't have filename, wtf"),
                                s,
                                t
                            );
                            println!("new filename is {}", new_filename);
                            render_settings.filename = Some(new_filename);
                        }
                        output_film(&render_settings, &film);
                    }
                }
                IntegratorType::LightTracing => {
                    let (bundled_settings, bundled_cameras): (Vec<RenderSettings>, Vec<Camera>) =
                        splatting_renders_and_cameras
                            .get(integrator_type)
                            .unwrap()
                            .iter()
                            .cloned()
                            .unzip();
                    let mut max_bounces = 0;
                    for settings in bundled_settings.iter() {
                        max_bounces = max_bounces.max(settings.max_bounces.unwrap_or(2));
                    }
                    let wavelength_bounds =
                        calculate_widest_wavelength_bounds(&bundled_settings, VISIBLE_RANGE);
                    world.assign_cameras(bundled_cameras.clone(), true);
                    if let EnvironmentMap::HDR {
                        texture,
                        importance_map,
                        ..
                    } = &mut world.environment
                    {
                        importance_map.bake_in_place(texture, wavelength_bounds);
                    }
                    let arc_world = Arc::new(world.clone());
                    let integrator = LightTracingIntegrator {
                        max_bounces,
                        world: arc_world.clone(),
                        russian_roulette: true,
                        camera_samples: 4,
                        wavelength_bounds,
                    };

                    println!("rendering with LightTracing integrator");
                    let render_splatted_result = NaiveRenderer::render_splatted(
                        integrator,
                        bundled_settings.clone(),
                        bundled_cameras.clone(),
                    );
                    assert!(render_splatted_result.len() > 0);
                    // films.extend(
                    //     (&bundled_settings)
                    //         .iter()
                    //         .cloned()
                    //         .zip(render_splatted_result),
                    // );
                    for (render_settings, film) in render_splatted_result {
                        output_film(&render_settings, &film);
                    }
                }

                _ => {}
            }
        }
    }
}
