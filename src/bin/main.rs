#![feature(fs_try_exists)]
extern crate rust_pathtracer as root;

use root::parsing::config::*;
use root::parsing::construct_world;
use root::parsing::get_settings;
use root::renderer::TiledRenderer;
use root::renderer::{NaiveRenderer, Renderer};

#[cfg(feature = "preview")]
use root::renderer::PreviewRenderer;
use root::world::*;

#[macro_use]
extern crate tracing;

use tracing::level_filters::LevelFilter;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use std::fs;
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
    pub scene: Option<String>,
    #[structopt(long, default_value = "data/config.toml")]
    pub config: String,
    #[structopt(short = "n", long)]
    pub dry_run: bool,
    #[structopt(short = "pll", long, default_value = "warn")]
    pub stdout_log_level: String,
    #[structopt(short = "wll", long, default_value = "info")]
    pub write_log_level: String,
}

fn construct_scene(config: &mut Config) -> anyhow::Result<World> {
    construct_world(config, PathBuf::from(config.scene_file.clone()))
}

fn construct_renderer(config: &Config) -> Box<dyn Renderer> {
    match config.renderer {
        RendererType::Naive { .. } => Box::new(NaiveRenderer::new()),
        #[cfg(feature = "preview")]
        RendererType::Preview { .. } => Box::new(PreviewRenderer::new()),
        RendererType::Tiled {
            tile_size: (width, height),
        } => Box::new(TiledRenderer::new(width, height)),
    }
}

fn parse_level_filter(level: String, default: LevelFilter) -> LevelFilter {
    match level.to_lowercase().as_str() {
        "warn" => LevelFilter::WARN,
        "info" => LevelFilter::INFO,
        "trace" => LevelFilter::TRACE,
        "error" => LevelFilter::ERROR,
        "debug" => LevelFilter::DEBUG,
        _ => default,
    }
}

fn main() {
    let opts = Opt::from_args();
    let stdout_log_level = parse_level_filter(opts.stdout_log_level, LevelFilter::WARN);
    let write_log_level = parse_level_filter(opts.write_log_level, LevelFilter::INFO);

    use std::{fs::File, sync::Arc};
    use tracing_subscriber::prelude::*;

    // A layer that logs events to stdout using the human-readable "pretty"
    // format.
    let stdout_log = tracing_subscriber::fmt::layer().pretty();

    // A layer that logs events to a file.
    let file = File::create("main.log").unwrap();
    let write_log = tracing_subscriber::fmt::layer().with_writer(Arc::new(file));

    tracing_subscriber::registry()
        .with(stdout_log.with_filter(stdout_log_level))
        .with(write_log.with_filter(write_log_level))
        .init();

    // // a builder for `FmtSubscriber`.
    // let subscriber = FmtSubscriber::builder()
    //     // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
    //     // will be written to stdout.
    //     .with_max_level(stdout_log_level)
    //     // completes the builder.
    //     .finish();

    // tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let mut cache_path = PathBuf::new();
    cache_path.push(".");
    cache_path.push("cache");
    if !cache_path.exists() {
        std::fs::DirBuilder::new()
            .create(cache_path)
            .expect("Failed to create cache directory. Does this process have permissions?");
    }

    let mut toml_config: TOMLConfig = match get_settings(opts.config) {
        Ok(expr) => expr,
        Err(v) => {
            error!("couldn't read config.toml, {:?}", v);

            return;
        }
    };

    let threads = toml_config
        .render_settings
        .iter()
        .map(|i| &i.threads)
        .fold(1, |a, &b| a.max(b.unwrap_or(1)));
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build_global()
        .unwrap();

    // override scene file based on provided command line argument
    toml_config.default_scene_file = opts.scene.unwrap_or(toml_config.default_scene_file);

    let mut config = Config::from(toml_config);
    let world = construct_scene(&mut config);
    if world.is_err() {
        error!(
            "fatal error parsing world, aborting. error is {:?}",
            world.err().unwrap()
        );
        return;
    }

    #[cfg(all(target_os = "windows", feature = "notification"))]
    let time = Instant::now();

    println!("constructing renderer");
    if !matches!(fs::try_exists("output"), Ok(true)) {
        fs::create_dir("output")
            .expect("failed to create output directory. please create it manually");
    }
    let renderer: Box<dyn Renderer> = construct_renderer(&config);

    if !opts.dry_run {
        renderer.render(world.unwrap(), &config);
        println!("render done");
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
