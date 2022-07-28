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
use medium::*;
use serde::de::DeserializeOwned;
use texture::*;

pub use curves::{
    load_csv, load_ior_and_kappa, load_linear, load_multiple_csv_rows,
    parse_tabulated_curve_from_csv, CurveData, CurveDataOrReference,
};

use crate::accelerator::AcceleratorType;
use crate::geometry::*;

use crate::materials::*;
use crate::mediums::MediumEnum;
use crate::texture::*;
// use crate::math::spectral::BOUNDED_VISIBLE_RANGE;
// use crate::math::*;
// use crate::world::EnvironmentMap;
use crate::world::World;

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use toml;

pub type Vec3Data = [f32; 3];
pub type Point3Data = [f32; 3];

#[derive(Serialize, Deserialize, Clone)]
pub enum MaybeCurvesLib {
    Literal(HashMap<String, CurveData>),
    Path(PathBuf),
}

impl MaybeCurvesLib {
    pub fn resolve(self) -> Result<HashMap<String, CurveData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(path),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum MaybeTextureLib {
    Literal(HashMap<String, TextureStackData>),
    Path(PathBuf),
}

impl MaybeTextureLib {
    pub fn resolve(self) -> Result<HashMap<String, TextureStackData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(path),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum MaybeMaterialLib {
    Literal(HashMap<String, MaterialData>),
    Path(PathBuf),
}

impl MaybeMaterialLib {
    pub fn resolve(self) -> Result<HashMap<String, MaterialData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(path),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum MaybeMediumLib {
    Literal(HashMap<String, MediumData>),
    Path(PathBuf),
}

impl MaybeMediumLib {
    pub fn resolve(self) -> Result<HashMap<String, MediumData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(path),
        }
    }
}

impl MaybeCurvesLib {}

#[derive(Serialize, Deserialize, Clone)]
pub struct SceneData {
    pub curves: MaybeCurvesLib,
    pub textures: MaybeTextureLib,
    pub materials: MaybeMaterialLib,
    pub mediums: MaybeMediumLib,
    pub instances: Vec<InstanceData>,
    pub environment: EnvironmentData,
    pub env_sampling_probability: Option<f32>,
}

// #[derive(Clone)]
// pub struct Scene {
//     pub textures: Vec<TexStack>,
//     pub materials: Vec<MaterialEnum>,
//     pub mediums: Vec<NamedMedium>,
//     pub instances: Vec<InstanceData>,
//     pub environment: EnvironmentData,
//     pub env_sampling_probability: f32,
// }

// impl From<SceneData> for Scene {
//     fn from(data: SceneData) -> Self {
//         let curves = data.curves.resolve();
//     }
// }

fn load_arbitrary<T>(filepath: PathBuf) -> Result<T, Box<dyn Error>>
where
    T: DeserializeOwned,
{
    let mut input = String::new();

    File::open(filepath.clone())
        .and_then(|mut f| f.read_to_string(&mut input))
        .expect(&format!("{}", filepath.to_string_lossy()).to_owned());

    let data: T = toml::from_str(&input)?;
    return Ok(data);
}

fn load_scene(filepath: PathBuf) -> Result<SceneData, Box<dyn Error>> {
    // will return Err in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    let result = File::open(filepath.clone()).and_then(|mut f| f.read_to_string(&mut input));

    if result.is_err() {
        error!("{}", filepath.to_string_lossy());
    }
    let read_count = result?;
    info!("read scene file: {} bytes", read_count);

    let scene: SceneData = toml::from_str(&input)?;
    return Ok(scene);
}

pub fn construct_world(scene_file: PathBuf) -> Result<World, Box<dyn Error>> {
    // parse scene from disk
    let scene = if let Ok(scene) = load_scene(scene_file) {
        scene
    } else {
        error!("failed to load scene");
        panic!()
    };

    // parse curves from disk or from literal
    let curves: HashMap<String, _> = if let Ok(curves) = scene.curves.resolve() {
        curves.into_iter().map(|(k, v)| (k, v.into())).collect()
    } else {
        error!("failed to parse curves");
        panic!()
    };

    // parse textures from disk or from literal
    let mut textures_map: HashMap<String, _> = HashMap::new();
    if let Ok(textures_data) = scene.textures.resolve() {
        for (name, data) in textures_data {
            if let Some(parsed) = parse_texture_stack(data, &curves) {
                textures_map.insert(name.clone(), parsed);
            } else {
                warn!("failed to parse texture {}", name);
            }
        }
    }

    // parse enviroment
    let environment = parse_environment(scene.environment, &textures_map);

    // parse materials from disk or from literal
    let mut materials_map: HashMap<String, _> = HashMap::new();
    if let Ok(materials_data) = scene.materials.resolve() {
        for (name, data) in materials_data {
            if let Some(parsed) = parse_material(data, &curves, &textures_map) {
                materials_map.insert(name.clone(), parsed);
            } else {
                warn!("failed to parse material {}", name);
            }
        }
    }

    // add mauve material to indicate errors.
    let mauve = MaterialEnum::DiffuseLight(DiffuseLight::new(
        crate::curves::mauve(1.0).to_cdf(math::spectral::EXTENDED_VISIBLE_RANGE, 20),
        math::Sidedness::Dual,
    ));

    // parse mediums from disk or from literal

    let mut mediums_map: HashMap<String, _> = HashMap::new();
    if let Ok(mediums_data) = scene.mediums.resolve() {
        for (name, data) in mediums_data {
            if let Some(parsed) = parse_medium(data, &curves) {
                mediums_map.insert(name.clone(), parsed);
            } else {
                warn!("failed to parse material {}", name);
            }
        }
    }

    // parse instances, and serialize all materials, mediums, and textures into vectors for storage in world.

    let mut materials: Vec<MaterialEnum> = Vec::new();
    let mut mediums: Vec<MediumEnum> = Vec::new();
    let mut instances: Vec<Instance> = Vec::new();
    let mut textures: Vec<TexStack> = Vec::new();

    let world = World::new(
        instances,
        materials,
        mediums,
        environment,
        scene.env_sampling_probability.unwrap_or(0.5),
        AcceleratorType::BVH,
    );
    Ok(world)

    // for tex in scene.textures {
    //     texture_count += 1;
    //     let tex_id = texture_count - 1;
    //     texture_names_to_ids.insert(tex.name.clone(), tex_id);

    //     info!("parsing texture {}", tex.name.clone());
    //     textures.push(parse_texture_stack(tex.clone()));
    // }

    // for material in scene.materials {
    //     let id = match material.data {
    //         MaterialData::DiffuseLight(_) | MaterialData::SharpLight(_) => {
    //             material_count += 1;
    //             MaterialId::Light((material_count - 1) as u16)
    //         }
    //         _ => {
    //             material_count += 1;
    //             MaterialId::Material((material_count - 1) as u16)
    //         }
    //     };
    //     material_names_to_ids.insert(material.name, id);
    //     info!("parsing material");
    //     materials.push(parse_material(
    //         material.data,
    //         &texture_names_to_ids,
    //         &textures,
    //     ));
    // }
    // if let Some(scene_mediums) = scene.mediums {
    //     for medium in scene_mediums {
    //         medium_count += 1;
    //         let id = medium_count - 1;
    //         medium_names_to_ids.insert(medium.name, id);
    //         info!("parsing medium");
    //         mediums.push(parse_medium(medium.data));
    //     }
    // }
    // for instance in scene.instances {
    //     match instance.aggregate {
    //         AggregateData::MeshBundle(data) => {
    //             let meshes = load_obj_file(&data.filename, &material_names_to_ids);
    //             let transform: Option<Transform3> = instance.transform.clone().map(|e| e.into());
    //             for mut mesh in meshes {
    //                 let id = instances.len();
    //                 mesh.init();
    //                 info!(
    //                     "pushing instance of mesh from meshbundle, with material id {:?} from {:?}",
    //                     instance
    //                         .material_identifier
    //                         .clone()
    //                         .map(|s| material_names_to_ids[&s]),
    //                     instance.material_identifier.clone()
    //                 );
    //                 instances.push(Instance::new(
    //                     Aggregate::Mesh(mesh),
    //                     transform,
    //                     instance
    //                         .material_identifier
    //                         .clone()
    //                         .map(|s| material_names_to_ids[&s]),
    //                     id,
    //                 ));
    //             }
    //         }
    //         _ => {
    //             info!("parsing instance and primitive");
    //             let id = instances.len();
    //             let instance = parse_instance(instance, &material_names_to_ids, id);
    //             instances.push(instance);
    //         }
    //     }
    // }
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn test_world() {
        let _world = construct_world(PathBuf::from("data/scenes/test_prism.toml")).unwrap();
    }
}
