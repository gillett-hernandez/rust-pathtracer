extern crate rust_pathtracer as root;

use log::LevelFilter;
use root::parsing::cameras::parse_config_and_cameras;
// use root::prelude::*;
use root::parsing::config::*;
use root::parsing::construct_world;
use root::parsing::get_settings;
use root::renderer::{NaiveRenderer, PreviewRenderer, Renderer};
use root::world::*;

#[macro_use]
extern crate log;
extern crate simplelog;

// use simplelog::*;
use simplelog::{ColorChoice, CombinedLogger, TermLogger, TerminalMode, WriteLogger};

use std::error::Error;
use std::fs::File;
use std::path::PathBuf;

use structopt::StructOpt;

#[cfg(all(target_os = "windows", feature = "notification"))]
use std::time::{Duration, Instant};
#[cfg(all(target_os = "windows", feature = "notification"))]
use win32_notification::NotificationBuilder;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(long)]
    pub scene_file: Option<String>,
    #[structopt(long, default_value = "data/config.toml")]
    pub config_file: String,
    #[structopt(short = "n", long)]
    pub dry_run: bool,
    #[structopt(short = "pll", long, default_value = "warn")]
    pub print_log_level: String,
    #[structopt(short = "wll", long, default_value = "info")]
    pub write_log_level: String,
}

fn construct_scene(config: &Config) -> Result<World, Box<dyn Error>> {
    construct_world(PathBuf::from(config.scene_file.clone()))
}

fn construct_renderer(config: &Config) -> Box<dyn Renderer> {
    match config.renderer {
        RendererType::Naive { .. } => Box::new(NaiveRenderer::new()),
        RendererType::Preview { .. } => Box::new(PreviewRenderer::new()),
    }
}

fn parse_log_level(level: String, default: LevelFilter) -> LevelFilter {
    match level.to_lowercase().as_str() {
        "warn" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "trace" => LevelFilter::Trace,
        "error" => LevelFilter::Error,
        "debug" => LevelFilter::Debug,
        _ => default,
    }
}

fn main() {
    let opts = Opt::from_args();
    let term_log_level = parse_log_level(opts.print_log_level, LevelFilter::Warn);
    let write_log_level = parse_log_level(opts.write_log_level, LevelFilter::Info);

    CombinedLogger::init(vec![
        TermLogger::new(
            term_log_level,
            simplelog::Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            write_log_level,
            simplelog::Config::default(),
            File::create("main.log").unwrap(),
        ),
    ])
    .unwrap();
    let mut config: TOMLConfig = match get_settings(opts.config_file) {
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
    config.default_scene_file = opts.scene_file.unwrap_or(config.default_scene_file);
    let (config, cameras) = parse_config_and_cameras(config);
    let world = construct_scene(&config);
    if world.is_err() {
        error!(
            "fatal error parsing world, aborting. error is {:?}",
            world.err().unwrap()
        );
        return;
    }

    #[cfg(all(target_os = "windows", feature = "notification"))]
    let time = Instant::now();
    let renderer: Box<dyn Renderer> = construct_renderer(&config);

    if !opts.dry_run {
        renderer.render(world.unwrap(), cameras, &config);
    }

    #[cfg(all(target_os = "windows", feature = "notification"))]
    {
        if opts.dry_run {
            // don't send notification if it's a dry run, since no rendering occurred
            return;
        }
        let notification = NotificationBuilder::new()
            .title_text("Render finished")
            .info_text(&format!("Took {} seconds", time.elapsed().as_secs()))
            .build()
            .expect("Could not create notification");

        notification.show().expect("Failed to show notification");
        std::thread::sleep(Duration::from_secs(3));
        notification
            .delete()
            .expect("Failed to delete notification");
    }
}
