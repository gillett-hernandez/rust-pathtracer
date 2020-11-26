extern crate rust_pathtracer as root;

use root::config::*;
use root::parsing::construct_world;
use root::renderer::{GPUStyleRenderer, NaiveRenderer, PreviewRenderer, Renderer};
use root::world::*;

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

    let renderer: Box<dyn Renderer> = construct_renderer(&config);
    renderer.render(world, cameras, &config);
}
