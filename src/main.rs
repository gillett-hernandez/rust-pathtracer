#![allow(unused_imports, unused_variables, unused)]

pub mod config;
pub mod integrator;
pub mod renderer;
pub mod world;

use config::{get_settings, RenderSettings, Settings};
use integrator::Integrator;
use renderer::{Film, Renderer};
use world::World;

use rayon::prelude::*;
// use std::error;
// use std::error::Error;
// use std::f32;
// use std::fmt;
// use std::env;
// use std::fs;

// use std::collections::HashMap;
// use std::fmt;
// use std::fs::File;
// use std::io::prelude::*;
// use std::io::BufReader;
// use std::sync::Arc;

fn render(renderer: &Renderer, film_settings: &RenderSettings) {
    let width = match film_settings.resolution {
        Some(res) => res.width,
        None => 512,
    };
    let height = match film_settings.resolution {
        Some(res) => res.height,
        None => 512,
    };
    println!(
        "starting render with film resolution {:?}x{:?}",
        width, height
    );
    let film = Film::new(width, height);
    renderer.render(&film, film_settings);
}

fn main() -> () {
    let config: Settings = match get_settings("data/config.toml".to_string()) {
        Ok(expr) => expr,
        Err(v) => {
            println!("{:?}", "couldn't read config.toml");
            println!("{:?}", v);
            return;
        }
    };
    assert!(config.output_directory != None);
    assert!(config.render_threads.unwrap() > 0);

    // let (renderer, integrator) = construct_renderer_and_integrator_from_config(config);
    // do_prerender_steps(config);

    // let cam_settings: &RenderSettings = &config.render_settings.unwrap()[0];
    // let integrator: Integrator = match (config.integrator) {
    //     Some(String::from("PT")) => PathTracingIntegrator(config),
    //     None => PathTracingIntegrator(config),
    // };
    let world = World { bvh: 0 };
    let renderer = Renderer { world };
    let settings_vec = config.render_settings.unwrap();
    // get settings for each film
    for film in settings_vec {
        // render(integrator, &cam_setting);
        render(&renderer, &film);
    }
}
