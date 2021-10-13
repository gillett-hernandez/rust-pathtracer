extern crate serde;

pub mod curves;
pub mod environment;
pub mod instance;
pub mod material;
pub mod medium;
pub mod primitives;
pub mod texture;

// use curves::*;
use environment::{parse_environment, EnvironmentData};
use instance::*;
use material::*;
use math::{Transform3, Vec3};
use medium::*;
use primitives::*;
use texture::*;

pub use curves::{
    load_csv, load_ior_and_kappa, load_linear, load_multiple_csv_rows,
    parse_tabulated_curve_from_csv,
};

use crate::{config::Config, mediums::MediumEnum};
// use crate::curves::*;
use crate::geometry::*;
use crate::materials::*;
use crate::texture::*;
// use crate::math::spectral::BOUNDED_VISIBLE_RANGE;
// use crate::math::*;
// use crate::world::EnvironmentMap;
use crate::world::{AcceleratorType, World};

use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;
use std::io::Read;

use serde::{Deserialize, Serialize};
use toml;

pub type Vec3Data = [f32; 3];
pub type Point3Data = [f32; 3];

#[derive(Serialize, Deserialize, Clone)]
pub struct Scene {
    pub textures: Vec<TextureStackData>,
    pub materials: Vec<NamedMaterial>,
    pub mediums: Option<Vec<NamedMedium>>,
    pub instances: Vec<InstanceData>,
    pub environment: EnvironmentData,
    pub env_sampling_probability: Option<f32>,
}

impl From<Transform3Data> for Transform3 {
    fn from(data: Transform3Data) -> Self {
        println!("parsing transform data");
        let maybe_scale = data.scale.map(|v| Transform3::from_scale(Vec3::from(v)));
        let maybe_rotate = if let Some(rotations) = data.rotate {
            let mut base = None;
            for rotation in rotations {
                let transform = Transform3::from_axis_angle(
                    Vec3::from(rotation.axis),
                    PI * rotation.angle / 180.0,
                );
                if base.is_none() {
                    base = Some(transform);
                } else {
                    base = Some(transform * base.unwrap());
                };
            }
            base
        } else {
            None
        };
        let maybe_translate = data
            .translate
            .map(|v| Transform3::from_translation(Vec3::from(v)));
        Transform3::from_stack(maybe_scale, maybe_rotate, maybe_translate)
    }
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
    let mut medium_names_to_ids: HashMap<String, usize> = HashMap::new();
    let mut texture_names_to_ids: HashMap<String, usize> = HashMap::new();
    let mut materials: Vec<MaterialEnum> = Vec::new();
    let mut mediums: Vec<MediumEnum> = Vec::new();
    let mut instances: Vec<Instance> = Vec::new();
    let mut textures: Vec<TexStack> = Vec::new();
    let mut material_count: usize = 0;
    let mut medium_count: usize = 0;
    let mut texture_count: usize = 0;
    for tex in scene.textures {
        texture_count += 1;
        let tex_id = texture_count - 1;
        texture_names_to_ids.insert(tex.name.clone(), tex_id);
        textures.push(parse_texture_stack(tex.clone()));
    }

    for material in scene.materials {
        let id = match material.data {
            MaterialData::DiffuseLight(_) | MaterialData::SharpLight(_) => {
                material_count += 1;
                MaterialId::Light((material_count - 1) as u16)
            }
            _ => {
                material_count += 1;
                MaterialId::Material((material_count - 1) as u16)
            }
        };
        material_names_to_ids.insert(material.name, id);
        materials.push(parse_material(
            material.data,
            &texture_names_to_ids,
            &textures,
        ));
    }
    if let Some(scene_mediums) = scene.mediums {
        for medium in scene_mediums {
            medium_count += 1;
            let id = medium_count - 1;
            medium_names_to_ids.insert(medium.name, id);
            mediums.push(parse_medium(medium.data));
        }
    }
    for instance in scene.instances {
        match instance.aggregate {
            AggregateData::MeshBundle(data) => {
                let meshes = load_obj_file(&data.filename, &material_names_to_ids);
                let transform: Option<Transform3> = instance.transform.clone().map(|e| e.into());
                for mut mesh in meshes {
                    let id = instances.len();
                    mesh.init();
                    println!(
                        "pushing instance of mesh from meshbundle, with material id {:?} from {:?}",
                        instance
                            .material_identifier
                            .clone()
                            .map(|s| material_names_to_ids[&s]),
                        instance.material_identifier.clone()
                    );
                    instances.push(Instance::new(
                        Aggregate::Mesh(mesh),
                        transform,
                        instance
                            .material_identifier
                            .clone()
                            .map(|s| material_names_to_ids[&s]),
                        id,
                    ));
                }
            }
            _ => {
                let id = instances.len();
                let instance = parse_instance(instance, &material_names_to_ids, id);
                instances.push(instance);
            }
        }
    }
    let world = World::new(
        instances,
        materials,
        mediums,
        parse_environment(scene.environment),
        scene.env_sampling_probability.unwrap_or(0.5),
        AcceleratorType::BVH,
    );
    world
}
