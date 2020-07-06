extern crate serde;

pub mod curves;
pub mod environment;
pub mod instance;
pub mod material;
pub mod primitives;

// use curves::*;
use environment::{parse_environment, EnvironmentData};
use instance::*;
use material::*;
// use primitives::*;

pub use curves::{
    load_csv, load_ior_and_kappa, load_linear, load_multiple_csv_rows,
    parse_tabulated_curve_from_csv,
};

use crate::config::Config;
// use crate::curves::*;
use crate::geometry::*;
use crate::materials::*;
// use crate::math::spectral::BOUNDED_VISIBLE_RANGE;
// use crate::math::*;
// use crate::world::EnvironmentMap;
use crate::world::{AcceleratorType, World};

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use serde::{Deserialize, Serialize};
use toml;

pub type Vec3Data = [f32; 3];
pub type Point3Data = [f32; 3];

#[derive(Serialize, Deserialize, Clone)]
pub struct Scene {
    pub instances: Vec<InstanceData>,
    pub materials: Vec<NamedMaterial>,
    pub environment: EnvironmentData,
    pub env_sampling_probability: Option<f32>,
}

fn get_scene(filepath: &str) -> Result<Scene, toml::de::Error> {
    // will return None in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    File::open(filepath)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();
    // uncomment the following line to print out the raw contents
    // println!("{:?}", input);
    let scene: Scene = toml::from_str(&input)?;
    // for render_settings in scene.render_settings.iter_mut() {
    //     render_settings.threads = match render_settings.threads {
    //         Some(expr) => Some(expr),
    //         None => Some(num_cpus as u16),
    //     };
    // }
    return Ok(scene);
}

pub fn construct_world(config: &Config) -> World {
    let scene = get_scene(&config.scene_file).expect("scene file failed to parse");
    let mut material_names_to_ids: HashMap<String, MaterialId> = HashMap::new();
    let mut materials: Vec<MaterialEnum> = Vec::new();
    let mut instances: Vec<Instance> = Vec::new();
    let mut material_count: usize = 0;
    for material in scene.materials {
        let id = match material.data {
            MaterialData::DiffuseLight(_) | MaterialData::SharpLight(_) => {
                material_count += 1;
                MaterialId::Light((material_count - 1) as u16)
            }
            MaterialData::Lambertian(_) | MaterialData::GGX(_) => {
                material_count += 1;
                MaterialId::Material((material_count - 1) as u16)
            }
        };
        material_names_to_ids.insert(material.name, id);
        materials.push(material.data.into());
    }
    for instance in scene.instances {
        let id = instances.len();
        instances.push(parse_instance(instance, &material_names_to_ids, id));
    }
    let world = World::new(
        instances,
        materials,
        parse_environment(scene.environment),
        scene.env_sampling_probability.unwrap_or(0.5),
        // TODO: switch this to bvh once triangles and meshes are implemented, since currently it causes a slowdown
        AcceleratorType::BVH,
        // AcceleratorType::List,
    );
    world
}
