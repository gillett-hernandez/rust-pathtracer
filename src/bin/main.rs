extern crate rust_pathtracer as root;
use math::*;
use root::camera::*;
use root::config::*;
use root::parsing::construct_world;
use root::renderer::{GPUStyleRenderer, NaiveRenderer, Renderer};
use root::world::*;

fn construct_scene(config: &Config) -> World {
    construct_world(config)
}

fn construct_renderer(config: &Config) -> Box<dyn Renderer> {
    match &*config.renderer {
        "Naive" => Box::new(NaiveRenderer::new()),
        "GPUStyle" => Box::new(GPUStyleRenderer::new()),
        _ => panic!(),
    }
}

fn main() -> () {
    let config: Config = match get_settings("data/config.toml".to_string()) {
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

    let world = construct_scene(&config);

    let cameras: Vec<Camera> = parse_cameras_from(&config);

    let renderer: Box<dyn Renderer> = construct_renderer(&config);
    renderer.render(world, cameras, &config);
}
