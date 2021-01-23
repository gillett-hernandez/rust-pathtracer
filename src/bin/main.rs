extern crate rust_pathtracer as root;

use root::config::*;
use root::parsing::construct_world;
use root::renderer::{GPUStyleRenderer, NaiveRenderer, PreviewRenderer, Renderer};
use root::world::*;

use std::time::Duration;
use std::{thread, time::Instant};
use win32_notification::NotificationBuilder;

fn construct_scene(config: &Config) -> World {
    construct_world(config)
}

fn construct_renderer(config: &Config) -> Box<dyn Renderer> {
    match config.renderer {
        RendererType::Naive { .. } => Box::new(NaiveRenderer::new()),
        RendererType::GPUStyle { .. } => Box::new(GPUStyleRenderer::new()),
        RendererType::Preview { .. } => Box::new(PreviewRenderer::new()),
    }
}

fn main() -> () {
    let config: TOMLConfig = match get_settings("data/config.toml".to_string()) {
        Ok(expr) => expr,
        Err(v) => {
            println!("{:?}", "couldn't read config.toml");
            println!("{:?}", v);
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

    let (config, cameras) = parse_cameras_from(&config);
    let world = construct_scene(&config);

    let time = Instant::now();
    let renderer: Box<dyn Renderer> = construct_renderer(&config);
    renderer.render(world, cameras, &config);

    let notification = NotificationBuilder::new()
        .title_text("Render finished")
        .info_text(&format!("Took {} seconds", time.elapsed().as_secs()))
        .build()
        .expect("Could not create notification");

    notification.show().expect("Failed to show notification");
    thread::sleep(Duration::from_secs(5));
    notification
        .delete()
        .expect("Failed to delete notification");
}
