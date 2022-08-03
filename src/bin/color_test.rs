extern crate rust_pathtracer as root;

use std::fs::File;

use math::curves::*;
use math::spectral::BOUNDED_VISIBLE_RANGE;
use math::*;

use root::parsing::{config::*, load_scene};
use root::renderer::Film;
use root::tonemap::Tonemapper;
use root::*;

#[macro_use]
extern crate log;
extern crate simplelog;

use crossbeam::channel::unbounded;
use log::LevelFilter;
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use simplelog::{ColorChoice, CombinedLogger, TermLogger, TerminalMode, WriteLogger};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(long, default_value = "data/config.toml")]
    pub config_file: String,
    #[structopt(long, default_value = "D65")]
    pub illuminant: String,
    pub width: usize,
    pub height: usize,

    pub bins: usize,
    pub ev_offset: f32,
}

fn main() {
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::Warn,
            simplelog::Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::Info,
            simplelog::Config::default(),
            File::create("main.log").unwrap(),
        ),
    ])
    .unwrap();

    let opts = Opt::from_args();
    let config: TOMLConfig = match get_settings(opts.config_file) {
        Ok(expr) => expr,
        Err(v) => {
            error!("couldn't read config.toml, {:?}", v);

            return;
        }
    };

    let threads = config
        .render_settings
        .iter()
        .map(|i| &i.threads)
        .fold(1, |a, &b| a.max(b.unwrap_or(1)));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global()
        .unwrap();

    // override scene file based on provided command line argument
    let (config, _) = parse_cameras_from(config);

    let curves = load_scene(config.scene_file)
        .expect("failed to load scene")
        .curves
        .resolve();

    let mut illuminants: Vec<Curve> = Vec::new();
    illuminants.push(curves.get("D65").unwrap().clone().into());
    illuminants.push(curves.get("E").unwrap().clone().into());
    illuminants.push(curves.get("cornell_light").unwrap().clone().into());
    illuminants.push(curves.get("cornell_light_accurate").unwrap().clone().into());
    illuminants.push(curves.get("blackbody_5000k").unwrap().clone().into());
    illuminants.push(curves.get("blackbody_3000k").unwrap().clone().into());
    illuminants.push(curves.get("fluorescent").unwrap().clone().into());
    illuminants.push(curves.get("xenon").unwrap().clone().into());

    let (width, height) = (opts.width, opts.height);

    let mut film = Film::new(width, height, XYZColor::BLACK);

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

    let (mut tonemapper, converter) = (
        // root::tonemap::Reinhard1x3::new(0.18, 1.0),
        root::tonemap::Reinhard0x3::new(0.18),
        root::tonemap::Converter::sRGB,
    );

    let bins = opts.bins;
    let ev_offset = opts.ev_offset;

    let color = Curve::Linear {
        signal: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        mode: InterpolationMode::Cubic,
        bounds: BOUNDED_VISIBLE_RANGE,
    };

    while window.is_open() && !window.is_key_pressed(Key::Escape, KeyRepeat::No) {
        film.buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, pixel)| {
                let (x, y) = (idx % width, idx / width);

                let bin_num = bins * x / width;
                let illuminant_bin = y * illuminants.len() / height;
                let illuminant = &illuminants[illuminant_bin];

                let strength = 1.1f32.powf(bin_num as f32 - ev_offset);
                // *pixel = color.to_xyz_color();
                let stacked = Curve::Machine {
                    seed: strength,
                    list: vec![(Op::Mul, color.clone()), (Op::Mul, illuminant.clone())],
                };
                *pixel = stacked.convert_to_xyz(BOUNDED_VISIBLE_RANGE, 1.0, false);
            });

        tonemapper.initialize(&film);
        buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel_idx, v)| {
                let y: usize = pixel_idx / film.width;
                let x: usize = pixel_idx - film.width * y;
                let [r, g, b, _]: [f32; 4] = converter
                    .transfer_function(tonemapper.map(&film, (x as usize, y as usize)), false)
                    .into();
                *v = rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
            });
        window
            .update_with_buffer(&buffer, film.width, film.height)
            .unwrap();
    }
}
