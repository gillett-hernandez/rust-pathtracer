extern crate rust_pathtracer as root;

use std::convert::Infallible;
use std::fs::File;
use std::sync::{Arc, RwLock};

use math::curves::*;
use math::spectral::BOUNDED_VISIBLE_RANGE;
use math::*;

use parking_lot::Mutex;
use root::parsing::{config::*, load_scene};
use root::renderer::Film;
use root::tonemap::{Converter, Tonemapper};
use root::*;

#[macro_use]
extern crate log;
extern crate simplelog;

use crossbeam::channel::{unbounded, Receiver, Sender};
use eframe::egui;
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
    pub width: usize,
    pub height: usize,
}

#[derive(Clone)]
enum Request {
    ChangeBins(usize),
    ChangeEvOffset(f32),
    ChangeColor(Curve),
    ChangeIlluminant(usize, Curve),
    ViewIlluminant(usize),
}

#[derive(Clone)]
enum Response {
    Illuminant(Option<Curve>),
}

struct Model {
    receiver: Receiver<Request>,
    sender: Sender<Response>,

    pub bins: usize,
    pub ev_offset: f32,
    pub illuminants: Vec<Curve>,
    pub color: Curve,
}

impl Model {
    pub fn new(
        response_sender: Sender<Response>,
        request_receiver: Receiver<Request>,
        bins: usize,
        ev_offset: f32,
        illuminants: Vec<Curve>,
        color: Curve,
    ) -> Self {
        Self {
            receiver: request_receiver,
            sender: response_sender,
            bins,
            ev_offset,
            illuminants,
            color,
        }
    }
    pub fn data_update(&mut self) {
        for request in self.receiver.try_iter() {
            match request {
                Request::ChangeBins(new) => self.bins = new,
                Request::ChangeEvOffset(new) => self.ev_offset = new,
                Request::ChangeColor(new) => self.color = new,
                Request::ChangeIlluminant(index, illuminant) => {
                    match self.illuminants.get_mut(index) {
                        Some(inner) => *inner = illuminant,
                        None => {}
                    }
                }
                Request::ViewIlluminant(index) => self
                    .sender
                    .try_send(Response::Illuminant(self.illuminants.get(index).cloned()))
                    .unwrap(),
            }
        }
    }
}

struct Controller {
    sender: Sender<Request>,
    receiver: Receiver<Response>,

    pub bins: usize,
    pub max_bins: usize,
    pub ev_offset: f32,
    pub ev_offset_bounds: Bounds1D,
    pub color: Curve,
    pub illuminant_index: usize,
    pub illuminant: Curve,
}

impl Controller {
    pub fn new(
        request_sender: Sender<Request>,
        response_receiver: Receiver<Response>,
        bins: usize,
        max_bins: usize,
        ev_offset: f32,
        ev_offset_range: Bounds1D,
        wavelength_bounds: Bounds1D,
    ) -> Self {
        Self {
            sender: request_sender,
            receiver: response_receiver,
            bins,
            max_bins,
            ev_offset,
            ev_offset_bounds: ev_offset_range,
            color: Curve::Linear {
                signal: vec![0.0],
                bounds: wavelength_bounds,
                mode: InterpolationMode::Cubic,
            },
            illuminant_index: 0,
            illuminant: Curve::Linear {
                signal: vec![0.0],
                bounds: wavelength_bounds,
                mode: InterpolationMode::Cubic,
            },
        }
    }
    pub fn data_update(&mut self) {
        for response in self.receiver.try_iter() {
            match response {
                Response::Illuminant(Some(illuminant)) => {
                    self.illuminant = illuminant;
                }
                _ => {}
            }
        }
    }
}

impl eframe::App for Controller {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.data_update();
            // let sender = &self.sender;

            let mut percentile = (100.0 * self.bins as f32 / self.max_bins as f32) as i32;
            let response = ui.add(egui::Slider::new(&mut percentile, 0..=100));
            if response.changed() {
                self.bins = (self.max_bins * percentile as usize) / 100;
                self.sender
                    .try_send(Request::ChangeBins(self.bins))
                    .unwrap()
            }

            let mut percentile = (100.0 * (self.ev_offset - self.ev_offset_bounds.lower)
                / self.ev_offset_bounds.span() as f32) as i32;
            let response = ui.add(egui::Slider::new(&mut percentile, 0..=100));
            if response.changed() {
                self.ev_offset = self.ev_offset_bounds.lower
                    + self.ev_offset_bounds.span() * percentile as f32 / 100.0;
                self.sender
                    .try_send(Request::ChangeEvOffset(self.ev_offset))
                    .unwrap()
            }
        });
    }
}

pub struct View {
    film: Film<XYZColor>,
    buffer: Vec<u32>,
    window: Window,
    tonemapper: Box<dyn Tonemapper>,
}

impl View {
    pub fn new(width: usize, height: usize, tonemapper: Box<dyn Tonemapper>) -> Self {
        let film = Film::new(width, height, XYZColor::BLACK);

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
        let buffer = vec![0u32; width * height];

        Self {
            film,
            buffer,
            window,
            tonemapper,
        }
    }

    fn update(&mut self, model: &Model) -> bool {
        let (width, height) = (self.film.width, self.film.height);
        self.film
            .buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, pixel)| {
                let illuminants = &model.illuminants;
                let (x, y) = (idx % width, idx / width);

                let bin_num = model.bins * x / width;
                let illuminant_bin = y * illuminants.len() / height;
                let illuminant = &illuminants[illuminant_bin];

                let strength = 1.1f32.powf(bin_num as f32 - model.ev_offset);
                // *pixel = color.to_xyz_color();
                let stacked = Curve::Machine {
                    seed: strength,
                    list: vec![
                        (Op::Mul, model.color.clone()),
                        (Op::Mul, illuminant.clone()),
                    ],
                };
                *pixel = stacked.convert_to_xyz(BOUNDED_VISIBLE_RANGE, 1.0, false);
            });

        self.tonemapper.initialize(&self.film);
        // let window = self.window
        let film = &self.film;
        let tonemapper = &self.tonemapper;
        self.buffer
            .par_iter_mut()
            .enumerate()
            .for_each(|(pixel_idx, v)| {
                let y: usize = pixel_idx / width;
                let x: usize = pixel_idx - width * y;
                let [r, g, b, _]: [f32; 4] = Converter::sRGB
                    .transfer_function(tonemapper.map(&film, (x as usize, y as usize)), false)
                    .into();
                *v = rgb_to_u32((256.0 * r) as u8, (256.0 * g) as u8, (256.0 * b) as u8);
            });
        // self.film = film;
        // self.tonemapper = tonemapper;
        self.window
            .update_with_buffer(&self.buffer, self.film.width, self.film.height)
            .unwrap();
        !self.window.is_open() || self.window.is_key_pressed(Key::Escape, KeyRepeat::No)
    }
}

fn mvc(opts: Opt) -> Result<(Model, Controller), ()>
// pub fn mvc<F>(opts: Opt) -> Result<(Model, F, Controller), ()>
// where
    // F: Fn() -> View,
{
    let config: TOMLConfig = match get_settings(opts.config_file) {
        Ok(expr) => expr,
        Err(v) => {
            error!("couldn't read config.toml, {:?}", v);

            return Err(());
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

    let mut illuminants = Vec::new();
    illuminants.push(curves.get("D65").unwrap().clone().into());
    illuminants.push(curves.get("E").unwrap().clone().into());
    illuminants.push(curves.get("cornell_light").unwrap().clone().into());
    illuminants.push(curves.get("cornell_light_accurate").unwrap().clone().into());
    illuminants.push(curves.get("blackbody_5000k").unwrap().clone().into());
    illuminants.push(curves.get("blackbody_3000k").unwrap().clone().into());
    illuminants.push(curves.get("fluorescent").unwrap().clone().into());
    illuminants.push(curves.get("xenon").unwrap().clone().into());

    let (req_sender, req_receiver) = unbounded();
    let (res_sender, res_receiver) = unbounded();

    // let (mut tonemapper, converter) = (
    //     // root::tonemap::Reinhard1x3::new(0.18, 1.0),
    //     root::tonemap::Reinhard0x3::new(0.18),
    //     root::tonemap::Converter::sRGB,
    // );

    let bins = 10;
    let ev_offset = -5.0;
    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
    let max_bins = 500;
    let ev_offset_range = Bounds1D::new(-50.0, 50.0);

    Ok((
        Model::new(
            res_sender,
            req_receiver,
            bins,
            ev_offset,
            illuminants,
            Curve::Linear {
                signal: vec![0.0, 0.0, 1.0, 0.0],
                bounds: wavelength_bounds,
                mode: InterpolationMode::Cubic,
            },
        ),
        // View::new(opts.width, opts.height, Box::new(tonemapper)),
        Controller::new(
            req_sender,
            res_receiver,
            bins,
            max_bins,
            ev_offset,
            ev_offset_range,
            wavelength_bounds,
        ),
    ))
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

    let (width, height) = (opts.width, opts.height);
    let (mut model, controller) = mvc(opts).expect("failed to construct MVC");

    // cannot join thread since run_native does not return
    let _ = std::thread::spawn(move || {
        let tonemapper = root::tonemap::Reinhard0x3::new(0.18);
        let mut view = View::new(width, height, Box::new(tonemapper));
        loop {
            model.data_update();
            let terminate = view.update(&model);
            if terminate {
                break;
            }
        }
    });

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(500.0, 900.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Show an image with eframe/egui",
        options,
        Box::new(|_cc| Box::new(controller)),
    );
}
