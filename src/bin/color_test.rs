extern crate rust_pathtracer as root;
use root::prelude::*;

use std::fs::File;
use std::ops::RangeInclusive;

use math::curves::*;

use root::parsing::config::TOMLConfig;
use root::parsing::parse_config_and_cameras;
use root::parsing::*;
use root::tonemap::{Converter, Tonemapper};

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
    #[structopt(long, default_value = "")]
    pub initial_color: String,
    #[structopt(long, default_value = "10.0")]
    pub dynamic_range: f32,
}

// TODO: add request and response, and impl for printing out CIE xy(Y) coordinates for colors as the illuminant strength increases
#[derive(Debug)]
enum Request {
    ChangeBins(usize),
    ChangeEvOffset(f32),
    ChangeColor(Curve),
    ChangeIlluminant(usize, Curve),
    AppendIlluminant(Curve),
    ViewIlluminant(usize),
    // IlluminantsCount,
}

#[derive(Debug)]
enum Response {
    Illuminant(Option<Curve>),
    // IlluminantsCount(usize),
}

struct Model {
    receiver: Receiver<Request>,
    sender: Sender<Response>,
    model_data: ModelData,
}

enum ModelData {
    Lightness(LightnessModel),
    Blending(BlendingModel),
}

struct LightnessModel {
    pub bins: usize,
    pub ev_multiplier: f32,
    pub ev_offset: f32,
    pub illuminants: Vec<Curve>,
    pub color: Curve,
    pub wavelength_bounds: Bounds1D,
}

struct BlendingModel {
    pub bins: usize,
    pub ev_multiplier: f32,
    pub ev_offset: f32,
    pub illuminants: Vec<Curve>,
    pub color: Curve,
    pub wavelength_bounds: Bounds1D,
}

impl Model {
    pub fn new(
        model_data: ModelData,
        response_sender: Sender<Response>,
        request_receiver: Receiver<Request>,
    ) -> Self {
        Model {
            model_data,
            receiver: request_receiver,
            sender: response_sender,
        }
    }

    pub fn data_update(&mut self) {
        for request in self.receiver.try_iter() {
            match self.model_data {
                ModelData::Lightness(lightness_data) => {
                    info!("{:?}", &request);
                    match request {
                        Request::ChangeBins(new) => lightness_data.bins = new,
                        Request::ChangeEvOffset(new) => lightness_data.ev_offset = new,
                        Request::ChangeColor(new) => {
                            println!("changed color");
                            lightness_data.color = new
                        }
                        Request::ChangeIlluminant(index, illuminant) => {
                            match lightness_data.illuminants.get_mut(index) {
                                Some(inner) => {
                                    println!("changed illuminant");
                                    *inner = illuminant
                                }
                                None => {
                                    println!("failed to change illuminant, index out of range");
                                }
                            }
                        }
                        Request::ViewIlluminant(index) => {
                            println!(
                                "got request, sending back illuminant {} from model (unless out of range)",
                                index
                            );
                            self.sender
                                .try_send(Response::Illuminant(lightness_data.illuminants.get(index).cloned()))
                                .unwrap()
                        }
                        Request::AppendIlluminant(new_illuminant) => {
                            lightness_data.illuminants.push(new_illuminant);
                        }
                        // Request::IlluminantsCount => self
                        //     .sender
                        //     .try_send(Response::IlluminantsCount(self.illuminants.len()))
                        //     .unwrap(),
                    }
                }
                ModelData::Blending(blending_data) => {
                    
                }
            }
        }
    }
}

impl LightnessModel {
    pub fn new(
        bins: usize,
        ev_multiplier: f32,
        ev_offset: f32,
        illuminants: Vec<Curve>,
        color: Curve,
        wavelength_bounds: Bounds1D,
    ) -> Self {
        Self {
            bins,
            ev_multiplier,
            ev_offset,
            illuminants,
            color,
            wavelength_bounds,
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
    pub illuminants_count: usize,
    pub wavelength_bounds: Bounds1D,
}

impl Controller {
    pub fn new(
        request_sender: Sender<Request>,
        response_receiver: Receiver<Response>,
        bins: usize,
        max_bins: usize,
        ev_offset: f32,
        ev_offset_range: Bounds1D,
        initial_color: Curve,
        illuminants_count: usize,
        wavelength_bounds: Bounds1D,
    ) -> Self {
        Self {
            sender: request_sender,
            receiver: response_receiver,
            bins,
            max_bins,
            ev_offset,
            ev_offset_bounds: ev_offset_range,
            color: initial_color,
            illuminant_index: 0,
            illuminant: Curve::Linear {
                signal: vec![0.0],
                bounds: wavelength_bounds,
                mode: InterpolationMode::Cubic,
            },
            illuminants_count,
            wavelength_bounds,
        }
    }
    pub fn data_update(&mut self) {
        // TODO: use rwlocks and Arcs to pass around curve data, instead of cloning them and sending them through channels
        for response in self.receiver.try_iter() {
            info!("{:?}", &response);

            match response {
                Response::Illuminant(Some(illuminant)) => {
                    println!("got illuminant from model");
                    self.illuminant = illuminant;
                }
                // Response::IlluminantsCount(num) => {
                //     self.illuminants_count = num;
                // }
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

            ui.label("number of bins");
            let response = ui.add(
                egui::DragValue::new(&mut self.bins)
                    .clamp_range(RangeInclusive::new(2, self.max_bins)),
            );
            if response.changed() {
                self.sender
                    .try_send(Request::ChangeBins(self.bins))
                    .unwrap()
            }

            ui.label("ev offset");
            let response = ui.add(egui::DragValue::new(&mut self.ev_offset).clamp_range(
                RangeInclusive::new(self.ev_offset_bounds.lower, self.ev_offset_bounds.upper),
            ));
            if response.changed() {
                self.sender
                    .try_send(Request::ChangeEvOffset(self.ev_offset))
                    .unwrap()
            }


            let response = ui.add(egui::Button::new("sync color"));
            if response.clicked() {
                println!("updating color data in model");
                self.sender
                    .try_send(Request::ChangeColor(self.color.clone()))
                    .unwrap()
            }

            // TODO: change this from index based to id based, both in the model and in this, and use dropdowns to select the illuminant.
            ui.label("illuminant index. select the index you'd like to view, then press 'view' to request it from the model");
            let _ = ui.add(
                egui::DragValue::new(&mut self.illuminant_index)
                    .clamp_range(RangeInclusive::new(0, self.illuminants_count + 1)),
            );

            let response = ui.add(egui::Button::new("view"));
            if response.clicked() {
                if self.illuminant_index < self.illuminants_count {
                    self.sender
                        .try_send(Request::ViewIlluminant(self.illuminant_index))
                        .unwrap()
                }
            }
            let response = ui.add(egui::Button::new("assign"));
            if response.clicked() {
                if self.illuminant_index < self.illuminants_count {
                    println!("changing illuminant {} to the illuminant in memory", self.illuminant_index);
                    self.sender
                        .try_send(Request::ChangeIlluminant(
                            self.illuminant_index,
                            self.illuminant.clone(),
                        ))
                        .unwrap()
                } else {
                    println!("appending a new illuminant");
                    self.sender
                        .try_send(Request::AppendIlluminant(self.illuminant.clone()))
                        .unwrap();
                    self.illuminants_count += 1;
                }
            }

            use egui::plot::{Line, Plot, PlotPoints};
            let n_samples = 100;
            let cloned = self.color.clone();
            let color = move |lambda: f64|{
                 cloned.evaluate(lambda as f32) as f64
            };
            let line = Line::new(PlotPoints::from_explicit_callback(color, (self.wavelength_bounds.lower as f64)..(self.wavelength_bounds.upper as f64), n_samples));

            let response = Plot::new("color")
                .include_x(self.wavelength_bounds.lower)
                .include_x(self.wavelength_bounds.upper)
                .include_y(0.0)
                .include_y(1.0)
                .view_aspect(2.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(line);
                    if plot_ui.plot_clicked() {
                        plot_ui.pointer_coordinate().map(|v| v.to_pos2())
                    } else {
                        None
                    }
                });

            let mut color_dirty = false;
            if let Some(clicked_point) = response.inner {
                if clicked_point.y >= 0.0 && self.wavelength_bounds.contains(&clicked_point.x) {
                    if let Curve::Tabulated { signal, .. } = &mut self.color {
                        let index = signal.partition_point(|(x, _)| *x < clicked_point.x);
                        signal.insert(index, (clicked_point.x, clicked_point.y.min(1.0)));
                        color_dirty = true;
                    }
                }
            }

            let response = ui.add(egui::Button::new("reset color"));
            if response.clicked() {
                if let Curve::Tabulated { signal, .. } = &mut self.color {
                    signal.clear();
                    signal.push((self.wavelength_bounds.lower, 0.0));
                    signal.push((self.wavelength_bounds.upper, 0.0));
                } else {
                    self.color = Curve::Tabulated {
                        signal: vec![
                            (self.wavelength_bounds.lower, 0.0),
                            (self.wavelength_bounds.upper, 0.0),
                        ],
                        mode: InterpolationMode::Cubic,
                    };
                }
                color_dirty = true;
            }

            if color_dirty {
                println!("updating color data in model");
                self.sender
                    .try_send(Request::ChangeColor(self.color.clone()))
                    .unwrap();
            }

            let cloned = self.illuminant.clone();
            let illuminant = move |lambda: f64| {
                 cloned.evaluate(lambda as f32) as f64
            };


            let line = Line::new(PlotPoints::from_explicit_callback(illuminant, (self.wavelength_bounds.lower as f64)..(self.wavelength_bounds.upper as f64), n_samples));
            Plot::new("illuminant")
                .include_x(self.wavelength_bounds.lower)
                .include_x(self.wavelength_bounds.upper)
                .include_y(0.0)
                .include_y(1.0)
                .view_aspect(2.0)
                .show(ui, |plot_ui| plot_ui.line(line));

            let new_temp_curve = Curve::Machine {
                seed: 1.0,
                list: vec![
                    (Op::Mul, self.color.clone()),
                    (Op::Mul, self.illuminant.clone()),
                ],
            };
            let multiplied = move |lambda| {
                new_temp_curve.evaluate(lambda as f32) as f64
            };
            // let line = Line::new(Values::from_values_iter(multiplied));
            let line = Line::new(PlotPoints::from_explicit_callback(multiplied, (self.wavelength_bounds.lower as f64)..(self.wavelength_bounds.upper as f64), n_samples));
            Plot::new("combined")
                .include_x(self.wavelength_bounds.lower)
                .include_x(self.wavelength_bounds.upper)
                .view_aspect(2.0)
                .show(ui, |plot_ui| plot_ui.line(line));
        });
    }
}

pub struct View {
    film: Film<XYZColor>,
    buffer: Vec<u32>,
    window: Window,
    tonemapper: Box<dyn Tonemapper>,
    per_illuminant_scale: Vec<f32>,
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
            per_illuminant_scale: vec![],
        }
    }

    fn update(&mut self, model: &LightnessModel) -> bool {
        let (width, height) = (self.film.width, self.film.height);

        for _ in self.per_illuminant_scale.len()..model.illuminants.len() {
            self.per_illuminant_scale.push(0.0);
        }
        for (i, illuminant) in model.illuminants.iter().enumerate() {
            let integral = illuminant.convert_to_xyz(model.wavelength_bounds, 1.0, false);
            // theoretically, this should not panic, since the above loop inserts empty elements when the lengths don't match
            self.per_illuminant_scale[i] = integral.y();
        }

        let per_illuminant_scale = &self.per_illuminant_scale;

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

                let strength = model.ev_multiplier.powf(bin_num as f32 - model.ev_offset);
                // *pixel = color.to_xyz_color();
                let stacked = Curve::Machine {
                    seed: strength / per_illuminant_scale[illuminant_bin],
                    list: vec![
                        (Op::Mul, model.color.clone()),
                        (Op::Mul, illuminant.clone()),
                    ],
                };
                *pixel = stacked.convert_to_xyz(model.wavelength_bounds, 1.0, false);
            });

        self.tonemapper.initialize(&self.film, 1.0);
        // let window = self.window
        // let film = &self.film;
        // let tonemapper = &self.tonemapper;
        update_window_buffer(
            &mut self.buffer,
            &self.film,
            self.tonemapper.as_mut(),
            Converter::sRGB,
            1.0,
        );
        self.window
            .update_with_buffer(&self.buffer, self.film.width, self.film.height)
            .unwrap();
        !self.window.is_open() || self.window.is_key_pressed(Key::Escape, KeyRepeat::No)
    }
}

fn mvc(opts: Opt) -> Result<(LightnessModel, Controller), ()> {
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
    let (config, _) = parse_config_and_cameras(config);

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

    let bins = 100;
    let ev_offset = (bins / 2) as f32;

    // log10(ev_multiplier ^ (bins/2) / ev_multiplier ^ (-bins/2)) = dynamic_range
    // log10(ev_multiplier^bins) = dynamic_range
    // bins * log10(ev_multiplier) = dynamic_range
    // ev_multiplier = 10^(dynamic_range / bins)
    assert!(opts.dynamic_range > 0.0);
    let ev_multiplier = 10.0f32.powf(opts.dynamic_range / bins as f32);
    let wavelength_bounds = BOUNDED_VISIBLE_RANGE;
    let max_bins = 200;
    let ev_offset_range = Bounds1D::new(-50.0, 250.0);
    let num_illuminants = illuminants.len();

    let initial_color = curves
        .get(&opts.initial_color)
        .cloned()
        .map(|e| e.into())
        .unwrap_or(Curve::Tabulated {
            signal: vec![(400.0, 1.0), (500.0, 0.2), (750.0, 1.0)],

            mode: InterpolationMode::Cubic,
        });

    Ok((
        LightnessModel::new(
            res_sender,
            req_receiver,
            bins,
            ev_multiplier,
            ev_offset,
            illuminants,
            initial_color.clone(),
            wavelength_bounds,
        ),
        // View::new(opts.width, opts.height, Box::new(tonemapper)),
        Controller::new(
            req_sender,
            res_receiver,
            bins,
            max_bins,
            ev_offset,
            ev_offset_range,
            initial_color,
            num_illuminants,
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
            File::create("color_test.log").unwrap(),
        ),
    ])
    .unwrap();

    let opts = Opt::from_args();

    let (width, height) = (opts.width, opts.height);
    let (mut model, controller) = mvc(opts).expect("failed to construct MVC");

    // cannot join thread since run_native does not return
    let _ = std::thread::spawn(move || {
        // let tonemapper = root::tonemap::Reinhard1x3::new(0.18, 1.0, true);
        let tonemapper = root::tonemap::Reinhard0x3::new(0.18, true);
        // let tonemapper = root::tonemap::Reinhard0::new(0.18, true);
        // let tonemapper = root::tonemap::Clamp::new(0.0, false, true);
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

    println!(
        " This applet allows you to view a color under multiple different illuminants.
you can also edit the color on the fly, if it is the proper type (Curve::Tabulated)
press reset curve to reset the curve to a flat zero tabulated curve that you can freely edit
adjust various sliders to change the EV offset.
"
    );
    eframe::run_native(
        "the same color under different illuminants",
        options,
        Box::new(|_cc| Box::new(controller)),
    );
}
