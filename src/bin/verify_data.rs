// checks a given data folder to confirm that files within it actually parse.

#![feature(fs_try_exists)]
extern crate rust_pathtracer as root;

use root::parsing::*;

use std::fs;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long, default_value = "data/config.toml")]
    path: String,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand, Clone, Copy, Default)]
enum Command {
    #[default]
    Config,
    Scene,
    Scenes,
    CurveLib,
    MaterialLib,
    MediumLib,
    MeshLib,
    TextureLib,
}

fn main() {
    let args = Args::parse();

    let Some(parse_type) = args.command else {
        return;
    };

    match parse_type {
        Command::Scene => {
            let maybe_scene = load_scene(args.path);
            match maybe_scene {
                Ok(_) => {}
                Err(e) => println!("{}", e.to_string()),
            }
        }
        Command::Scenes => {
            let Ok(dir) = fs::read_dir(&args.path) else {
                println!("failed to load text from path {}", &args.path);
                return;
            };

            for entry in dir.filter_map(|e| e.ok()) {
                let filename_osstr = entry.file_name();
                let filename = filename_osstr.to_string_lossy();

                if filename.ends_with(".toml") {
                    let maybe_scene = load_scene(entry.path());
                    match maybe_scene {
                        Ok(_) => {}
                        Err(e) => println!("{}", e.to_string()),
                    }
                }
            }
        }
        Command::Config => {
            let maybe_config = get_config(args.path);
            match maybe_config {
                Ok(config) => {
                    println!("renderer {:?}", config.renderer);
                }
                Err(e) => println!("{}", e.to_string()),
            }
        }
        Command::CurveLib => {
            let Ok(data) = fs::read_to_string(&args.path) else {
                println!("failed to load text from path {}", &args.path);
                return;
            };
            let res: Result<MaybeCurveLib, _> = toml::from_str(&data);
            match res {
                Ok(lib) => {
                    let resolved = lib.resolve();
                    for (key, value) in &resolved {
                        println!("found {} - {:?} pair", key, value);
                    }
                }
                Err(failure) => {
                    println!("{}", failure.to_string());
                }
            }
        }
        Command::MaterialLib => {
            let Ok(data) = fs::read_to_string(args.path) else {
                println!("failed to load text from path");
                return;
            };
            let res: Result<MaybeMaterialLib, _> = toml::from_str(&data);
            match res {
                Ok(lib) => {
                    let resolved = lib.resolve();
                    for (key, value) in &resolved {
                        println!("found {} - {:?} pair", key, value);
                    }
                }
                Err(failure) => {
                    println!("{}", failure.to_string());
                }
            }
        }
        Command::MediumLib => {
            let Ok(data) = fs::read_to_string(args.path) else {
                println!("failed to load text from path");
                return;
            };
            let res: Result<MaybeMediumLib, _> = toml::from_str(&data);
            match res {
                Ok(lib) => {
                    let resolved = lib.resolve();
                    for (key, value) in &resolved {
                        println!("found {} - {:?} pair", key, value);
                    }
                }
                Err(failure) => {
                    println!("{}", failure.to_string());
                }
            }
        }
        Command::MeshLib => {
            let Ok(data) = fs::read_to_string(args.path) else {
                println!("failed to load text from path");
                return;
            };
            let res: Result<MaybeMeshLib, _> = toml::from_str(&data);
            match res {
                Ok(lib) => {
                    let resolved = lib.resolve();
                    for (key, value) in &resolved {
                        println!("found {} - {:?} pair", key, value);
                    }
                }
                Err(failure) => {
                    println!("{}", failure.to_string());
                }
            }
        }
        Command::TextureLib => {
            let Ok(data) = fs::read_to_string(args.path) else {
                println!("failed to load text from path");
                return;
            };
            let res: Result<MaybeTextureStackLib, _> = toml::from_str(&data);
            match res {
                Ok(lib) => {
                    let resolved = lib.resolve();
                    for (key, value) in &resolved {
                        println!("found {} - {:?} pair", key, value);
                    }
                }
                Err(failure) => {
                    println!("{}", failure.to_string());
                }
            }
        }
    }
}
