#![allow(unused_imports, unused_variables, unused)]
extern crate num_cpus;
extern crate serde;

use std::env;
use std::fs::File;
use std::io::Read;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use toml;

#[derive(Deserialize, Copy, Clone)]

pub struct Resolution {
    pub width: usize,
    pub height: usize,
}

#[derive(Deserialize, Copy, Clone)]
pub struct RenderSettings {
    pub resolution: Option<Resolution>,
    pub min_samples: Option<u16>,
    pub max_samples: Option<u16>,
    pub camera_id: Option<u16>,
}

#[derive(Deserialize, Clone)]
pub struct Settings {
    pub output_directory: Option<String>,
    pub integrator: Option<String>,
    pub render_threads: Option<i16>,
    pub render_settings: Option<Vec<RenderSettings>>,
}

pub fn get_settings(filepath: String) -> Result<Settings, toml::de::Error> {
    // will return None in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    File::open(&filepath)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();
    // uncomment the following line to print out the raw contents
    // println!("{:?}", input);
    let mut settings: Settings = toml::from_str(&input)?;
    let num_cpus = num_cpus::get();
    settings.render_threads = match settings.render_threads {
        Some(expr) => Some(expr),
        None => Some(num_cpus as i16),
    };
    return Ok(settings);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parsing_config() {
        let settings: Settings = match get_settings("data/config.toml".to_string()) {
            Ok(expr) => expr,
            Err(v) => {
                println!("{:?}", "couldn't read config.toml");
                println!("{:?}", v);
                return;
            }
        };
        assert!(settings.output_directory != None);
        assert!(settings.render_threads.unwrap() > 0)
    }
}
