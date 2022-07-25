extern crate rust_pathtracer as root;

use log::LevelFilter;
use root::config::*;
use root::parsing::construct_world;
use root::renderer::{NaiveRenderer, PreviewRenderer, Renderer};
use root::world::*;

#[macro_use]
extern crate log;
extern crate simplelog;

// use simplelog::*;
use simplelog::{ColorChoice, CombinedLogger, TermLogger, TerminalMode, WriteLogger};

use std::fs::File;

use structopt::StructOpt;

#[cfg(all(target_os = "windows", feature = "notification"))]
use std::time::Instant;
#[cfg(all(target_os = "windows", feature = "notification"))]
use win32_notification::NotificationBuilder;

#[derive(Debug, StructOpt)]
#[structopt(rename_all = "kebab-case")]
struct Opt {
    #[structopt(long)]
    pub scene_file: Option<String>,
    #[structopt(long, default_value = "data/config.toml")]
    pub config_file: String,
}

fn construct_scene(config: &Config) -> World {
    construct_world(&config.scene_file)
}

fn construct_renderer(config: &Config) -> Box<dyn Renderer> {
    match config.renderer {
        RendererType::Naive { .. } => Box::new(NaiveRenderer::new()),
        RendererType::Preview { .. } => Box::new(PreviewRenderer::new()),
    }
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
    let (config, cameras) = parse_cameras_from(&config);
    let world = construct_scene(&config);

    #[cfg(all(target_os = "windows", feature = "notification"))]
    let time = Instant::now();
    let renderer: Box<dyn Renderer> = construct_renderer(&config);
    renderer.render(world, cameras, &config);

    #[cfg(all(target_os = "windows", feature = "notification"))]
    {
        let notification = NotificationBuilder::new()
            .title_text("Render finished")
            .info_text(&format!("Took {} seconds", time.elapsed().as_secs()))
            .build()
            .expect("Could not create notification");

        notification.show().expect("Failed to show notification");
        thread::sleep(Duration::from_secs(3));
        notification
            .delete()
            .expect("Failed to delete notification");
    }
}
