use super::prelude::*;
use crate::prelude::*;

use crate::integrator::*;

use math::spectral::BOUNDED_VISIBLE_RANGE as VISIBLE_RANGE;

use std::collections::HashMap;
use std::ops::Deref;
use std::ops::DerefMut;
// use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::unbounded;
// use crossbeam::channel::{bounded};
use pbr::ProgressBar;
use rayon::iter::ParallelIterator;

#[derive(Default, Copy, Clone)]
pub struct TiledRenderer {
    tile_size: (u16, u16),
}

#[derive(Clone, Copy, Debug, Hash, Default)]
pub enum TileStatus {
    #[default]
    Incomplete,
    InProgress,
    CompletedButNotSynced,
    Complete,
}

fn iter_pixels(
    tile: impl Deref<Target = ((u16, u16), (u16, u16))>,
) -> impl Iterator<Item = (u16, u16)> {
    (tile.0 .0..tile.0 .1)
        .into_iter()
        .flat_map(move |x| (tile.1 .0..tile.1 .1).map(move |y| (x, y)))
}

impl TiledRenderer {
    pub fn new(tile_width: u16, tile_height: u16) -> TiledRenderer {
        warn!("constructing tiled renderer");
        TiledRenderer {
            tile_size: (tile_width, tile_height),
        }
    }

    pub fn generate_tiles(&self, film_size: (usize, usize)) -> Vec<((u16, u16), (u16, u16))> {
        let tile_size = self.tile_size;
        let full_tile_count = (
            film_size.0 / tile_size.0 as usize,
            film_size.1 / tile_size.1 as usize,
        );
        let remnant_tile_size = (
            film_size.0 % tile_size.0 as usize,
            film_size.1 % tile_size.1 as usize,
        );

        let mut tiles = Vec::new();
        for y_idx in 0..full_tile_count.1 {
            for x_idx in 0..full_tile_count.0 {
                tiles.push((
                    (
                        x_idx as u16 * tile_size.0,
                        x_idx as u16 * tile_size.0 + tile_size.0,
                    ),
                    (
                        y_idx as u16 * tile_size.1,
                        y_idx as u16 * tile_size.1 + tile_size.1,
                    ),
                ));
            }
        }
        if remnant_tile_size.0 > 0 {
            // add all right side partial tiles
            for y_idx in 0..full_tile_count.1 {
                tiles.push((
                    (
                        full_tile_count.0 as u16 * tile_size.0,
                        full_tile_count.0 as u16 * tile_size.0 + remnant_tile_size.0 as u16,
                    ),
                    (
                        y_idx as u16 * tile_size.1,
                        y_idx as u16 * tile_size.1 + tile_size.1,
                    ),
                ));
            }
        }
        if remnant_tile_size.1 > 0 {
            // add all bottom side partial tiles
            for x_idx in 0..full_tile_count.0 {
                tiles.push((
                    (
                        x_idx as u16 * tile_size.0,
                        x_idx as u16 * tile_size.0 + tile_size.0,
                    ),
                    (
                        full_tile_count.1 as u16 * tile_size.1,
                        full_tile_count.1 as u16 * tile_size.1 + remnant_tile_size.1 as u16,
                    ),
                ));
            }
            if remnant_tile_size.0 > 0 {
                // both
                // add last partial tile at bottom right
                tiles.push((
                    (
                        full_tile_count.0 as u16 * tile_size.0,
                        full_tile_count.0 as u16 * tile_size.0 + remnant_tile_size.0 as u16,
                    ),
                    (
                        full_tile_count.1 as u16 * tile_size.1,
                        full_tile_count.1 as u16 * tile_size.1 + remnant_tile_size.1 as u16,
                    ),
                ))
            }
        }
        tiles
    }

    pub fn render_sampled<I: SamplerIntegrator>(
        self,
        mut integrator: I,
        settings: &RenderSettings,
        _camera: &CameraEnum,
    ) -> Film<XYZColor> {
        let (width, height) = (settings.resolution.width, settings.resolution.height);
        warn!("starting render with film resolution {}x{}", width, height);
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
        integrator.preprocess(&mut presampler, &[settings.clone()], &mut preprofile);

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

        let tiles = self.generate_tiles((width, height));

        let ptr = DoNotDoThisEver(film.buffer.as_mut_ptr());

        let stats: Profile = tiles
            .par_iter()
            .map(|tile| {
                let mut profile = Profile::default();

                // let mut sampler: Box<dyn Sampler> = Box::new(StratifiedSampler::new(20, 20, 10));
                let mut sampler: Box<dyn Sampler> = Box::new(RandomSampler::new());
                let mut ct = 0;
                for (x, y) in iter_pixels(tile) {
                    let mut temp_color = XYZColor::BLACK;

                    for s in 0..settings.min_samples {
                        let sample = sampler.draw_2d();

                        // box filter
                        let camera_uv = (
                            (x as f32 + sample.x) / (settings.resolution.width as f32),
                            (y as f32 + sample.y) / (settings.resolution.height as f32),
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
                    }

                    unsafe {
                        *ptr.0.add(y as usize * width + x as usize) =
                            temp_color / (settings.min_samples as f32);
                    }
                    ct += 1;
                }
                pixel_count.fetch_add(ct, Ordering::Relaxed);

                profile
            })
            .reduce(Profile::default, |a, b| a.combine(b));

        if let Err(panic) = thread.join() {
            println!(
                "progress bar incrememnting thread threw an error {:?}",
                panic
            );
        }
        println!();
        let elapsed = (now.elapsed().as_millis() as f32) / 1000.0;
        println!("took {}s", elapsed);
        stats.pretty_print(elapsed, settings.threads.unwrap() as usize);
        film
    }
    pub fn render_splatted<I: GenericIntegrator>(
        mut integrator: I,
        renders: Vec<RenderSettings>,
        _cameras: Vec<CameraEnum>,
    ) -> Vec<(RenderSettings, Film<XYZColor>)> {
        vec![]
    }
}

impl Renderer for TiledRenderer {
    fn render(&self, mut world: World, cameras: Vec<CameraEnum>, config: &Config) {
        // bin the render settings into bins corresponding to what integrator they need.

        let mut bundled_cameras: Vec<CameraEnum> = Vec::new();
        // let mut films: Vec<(RenderSettings, Film<XYZColor>)> = Vec::new();
        let mut sampled_renders: Vec<(IntegratorType, RenderSettings)> = Vec::new();
        let mut splatting_renders_and_cameras: HashMap<
            IntegratorType,
            Vec<(RenderSettings, CameraEnum)>,
        > = HashMap::new();
        // splatting_renders_and_cameras.insert(IntegratorType::BDPT, Vec::new());
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
            let copied_camera = cameras[camera_id].clone().with_aspect_ratio(aspect_ratio);

            let integrator_type: IntegratorType = IntegratorType::from(render_settings.integrator);

            match integrator_type {
                IntegratorType::PathTracing => {
                    let mut updated_render_settings = render_settings.clone();
                    updated_render_settings.camera_id = camera_id;
                    bundled_cameras.push(copied_camera);
                    sampled_renders.push((IntegratorType::PathTracing, updated_render_settings));
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
                IntegratorType::PathTracing => {
                    world.assign_cameras(vec![cameras[render_settings.camera_id].clone()], false);

                    let env_sampling_probability = world.get_env_sampling_probability();
                    if let EnvironmentMap::HDR {
                        texture,
                        importance_map,
                        strength,
                        ..
                    } = &mut world.environment
                    {
                        if *strength > 0.0 && env_sampling_probability > 0.0 {
                            let wavelength_bounds = render_settings
                                .wavelength_bounds
                                .map(|e| Bounds1D::new(e.0, e.1))
                                .unwrap_or(math::spectral::BOUNDED_VISIBLE_RANGE);
                            importance_map.bake_in_place(texture, wavelength_bounds);
                        }
                    }
                    let arc_world = Arc::new(world.clone());

                    if let Some(Integrator::PathTracing(integrator)) =
                        Integrator::from_settings_and_world(
                            arc_world.clone(),
                            IntegratorType::PathTracing,
                            &bundled_cameras,
                            render_settings,
                        )
                    {
                        warn!("rendering with PathTracing integrator");
                        let (render_settings, film) = (
                            render_settings.clone(),
                            self.render_sampled(
                                integrator,
                                render_settings,
                                &cameras[render_settings.camera_id],
                            ),
                        );
                        output_film(&render_settings, &film, 1.0);
                    }
                }
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
                IntegratorType::LightTracing => {
                    unimplemented!()
                }

                _ => {}
            }
        }
    }
}

struct DoNotDoThisEver<T>(*mut T);
unsafe impl Send for DoNotDoThisEver<f32> {}
unsafe impl Sync for DoNotDoThisEver<f32> {}
unsafe impl Send for DoNotDoThisEver<f32x4> {}
unsafe impl Sync for DoNotDoThisEver<f32x4> {}
unsafe impl Send for DoNotDoThisEver<XYZColor> {}
unsafe impl Sync for DoNotDoThisEver<XYZColor> {}

#[cfg(test)]
mod test {
    use minifb::WindowOptions;
    use rand::random;

    use crate::tonemap::Clamp;

    use super::*;
    #[test]
    fn test_generate_tiles() {
        let mut film = Film::new(1920, 1080, false);
        let renderer = TiledRenderer::new(64, 64);

        let tiles = renderer.generate_tiles((film.width, film.height));
        for tile in tiles {
            for pixel in iter_pixels(&tile) {
                film.write_at(pixel.0 as usize, pixel.1 as usize, true);
            }
        }

        assert!(film.buffer.iter().all(|e| *e));
    }

    #[test]
    fn test_parallel_unsafe_access() {
        let mut film = Film::new(1920, 1080, 0.0f32);
        let renderer = TiledRenderer::new(64, 64);

        let tiles = renderer.generate_tiles((film.width, film.height));
        let width = film.width;
        let ptr = DoNotDoThisEver(film.buffer.as_mut_ptr());
        tiles.par_iter().for_each(|tile| {
            for pixel in iter_pixels(tile) {
                unsafe {
                    *ptr.0.add(pixel.1 as usize * width + pixel.0 as usize) += 0.25;
                }
            }
        });

        let expected = 1920.0 * 1080.0 * 0.25;
        // assert_eq!(film.buffer.iter().cloned().sum::<f32>(), expected);
        println!(
            "{} == {}?",
            film.buffer.iter().cloned().sum::<f32>(),
            expected
        );

        let new_film = Film {
            buffer: film
                .buffer
                .iter()
                .cloned()
                .map(|e| SingleWavelength::new(550.0, e).into())
                .collect::<Vec<XYZColor>>(),
            width,
            height: film.height,
        };

        let mut tonemapper = Clamp::new(0.0, true, true);
        window_loop(
            1920,
            1080,
            144,
            WindowOptions {
                borderless: true,
                ..Default::default()
            },
            true,
            |window, mut window_buffer, width, height| {
                update_window_buffer(
                    &mut window_buffer,
                    &new_film,
                    &mut tonemapper,
                    Converter::sRGB,
                    1.0,
                );
            },
        )
    }

    #[test]
    fn test_viewing_while_writing_unsafe_parallel() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(16 as usize)
            .build_global()
            .unwrap();

        let mut film = Film::new(1920, 1080, XYZColor::BLACK);
        let renderer = TiledRenderer::new(32, 32);
        let num_samples = 1000;

        let tiles = renderer.generate_tiles((film.width, film.height));
        let width = film.width;
        let height = film.height;
        // let capacity = film.buffer.capacity();
        let ptr = DoNotDoThisEver(film.buffer.as_mut_ptr());
        let tile_status = (0..tiles.len())
            .map(|_| RwLock::new(TileStatus::Incomplete))
            .collect::<Vec<RwLock<TileStatus>>>();

        thread::scope(|s| {
            let join_handle = s.spawn(|| {
                tiles.par_iter().enumerate().for_each(|(tile_index, tile)| {
                    {
                        let mut locked = tile_status[tile_index].write().unwrap();
                        *locked = TileStatus::InProgress;
                    }

                    for pixel in iter_pixels(tile) {
                        for _ in 0..num_samples {
                            let wavelength = BOUNDED_VISIBLE_RANGE.sample(random::<f32>()) ;
                            let energy = 1.0;
                            let center = Vec3::new(width as f32 / 2.0, height as f32 / 2.0, 0.0);
                            let dist_squared = (Vec3::new(pixel.0 as f32, pixel.1 as f32, 0.0)
                                - center)
                                .norm_squared();
                            let diffraction_mult = (dist_squared / wavelength).cos().powi(2);

                            unsafe {
                                *ptr.0.add(pixel.1 as usize * width + pixel.0 as usize) +=
                                    XYZColor::from(SingleWavelength::new(
                                        wavelength,
                                        diffraction_mult * energy
                                            / (1.0
                                                + 100.0
                                                    * (dist_squared / (wavelength * wavelength)))
                                            / num_samples as f32,
                                    ));
                            }
                        }
                    }
                    {
                        let mut locked = tile_status[tile_index].write().unwrap();
                        *locked = TileStatus::CompletedButNotSynced;
                    }
                })
            });

            println!("point 1");
            let mut tonemapper = Clamp::new(0.0, true, true);

            window_loop(
                1920,
                1080,
                2,
                WindowOptions {
                    borderless: true,
                    ..Default::default()
                },
                false,
                |window, mut window_buffer, width, height| {
                    // extremely unsafe, do not do this

                    println!("loop");

                    let width = film.width;
                    debug_assert!(window_buffer.len() % width == 0);
                    tonemapper.initialize(&film, 1.0);

                    // local copy of status's

                    let copy = tile_status
                        .iter()
                        .map(|e| e.read().unwrap().clone())
                        .collect::<Vec<TileStatus>>();

                    for (i, (tile, single_tile_status)) in tiles.iter().zip(copy.iter()).enumerate()
                    {
                        match single_tile_status {
                            status @ TileStatus::InProgress
                            | status @ TileStatus::CompletedButNotSynced => {
                                for (x, y) in iter_pixels(tile) {
                                    let [r, g, b, _]: [f32; 4] = Converter::sRGB
                                        .transfer_function(
                                            tonemapper.map(&film, (x as usize, y as usize)),
                                            false,
                                        )
                                        .into();
                                    window_buffer[y as usize * width + x as usize] = rgb_to_u32(
                                        (256.0 * r) as u8,
                                        (256.0 * g) as u8,
                                        (256.0 * b) as u8,
                                    );
                                }
                                if matches!(status, TileStatus::CompletedButNotSynced) {
                                    *tile_status[i].write().unwrap() = TileStatus::Complete;
                                }
                            }
                            _ => {}
                        }
                    }
                },
            );
            let _ = join_handle.join();
        });
    }
}
