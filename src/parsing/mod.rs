extern crate serde;

pub mod config;
pub mod curves;
pub mod environment;
pub mod instance;
pub mod material;
pub mod medium;
pub mod primitives;
pub mod texture;
pub mod tonemap;

use environment::{parse_environment, EnvironmentData};
use instance::*;
use material::*;
use math::Transform3;
use medium::*;
use serde::de::DeserializeOwned;
use texture::*;
pub use tonemap::parse_tonemapper;

pub use curves::{
    load_csv, load_ior_and_kappa, load_linear, load_multiple_csv_rows,
    parse_tabulated_curve_from_csv, CurveData, CurveDataOrReference,
};

use crate::accelerator::AcceleratorType;
use crate::geometry::*;

use crate::materials::*;
use crate::mediums::MediumEnum;
use crate::world::World;

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use toml;

use self::primitives::load_obj_file;
use self::primitives::AggregateData;

pub type Vec3Data = [f32; 3];
pub type Point3Data = [f32; 3];

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MaybeCurvesLib {
    Literal(HashMap<String, CurveData>),
    Path(String),
}

impl MaybeCurvesLib {
    pub fn resolve(self) -> Result<HashMap<String, CurveData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(PathBuf::from(path)),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MaybeTextureLib {
    Literal(HashMap<String, TextureStackData>),
    Path(String),
}

impl MaybeTextureLib {
    pub fn resolve(self) -> Result<HashMap<String, TextureStackData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(PathBuf::from(path)),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MaybeMaterialLib {
    Literal(HashMap<String, MaterialData>),
    Path(String),
}

impl MaybeMaterialLib {
    pub fn resolve(self) -> Result<HashMap<String, MaterialData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(PathBuf::from(path)),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MaybeMediumLib {
    Literal(HashMap<String, MediumData>),
    Path(String),
}

impl MaybeMediumLib {
    pub fn resolve(self) -> Result<HashMap<String, MediumData>, Box<dyn Error>> {
        match self {
            Self::Literal(data) => Ok(data),
            Self::Path(path) => load_arbitrary(PathBuf::from(path)),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SceneData {
    pub env_sampling_probability: Option<f32>,
    pub environment: EnvironmentData,
    pub curves: MaybeCurvesLib,
    pub textures: MaybeTextureLib,
    pub materials: MaybeMaterialLib,
    pub mediums: Option<MaybeMediumLib>,
    pub instances: Vec<InstanceData>,
}

fn load_arbitrary<T>(filepath: PathBuf) -> Result<T, Box<dyn Error>>
where
    T: DeserializeOwned,
{
    info!("loading file at {}", &filepath.to_string_lossy());
    let mut input = String::new();

    File::open(filepath.clone())
        .and_then(|mut f| f.read_to_string(&mut input))
        .expect("failed to load file");

    let data: T = toml::from_str(&input)?;
    return Ok(data);
}

fn load_scene(filepath: PathBuf) -> Result<SceneData, Box<dyn Error>> {
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    let result = File::open(filepath.clone()).and_then(|mut f| f.read_to_string(&mut input));

    info!("loading file, {}", filepath.to_string_lossy());
    let read_count = result.inspect_err(|e| {
        error!("{}", e.to_string());
    })?;

    info!("done: {} bytes", read_count);

    let scene: SceneData = toml::from_str(&input).inspect_err(|e| {
        error!(
            "encountered error when parsing scene file: {}",
            e.to_string()
        );
        if let Some(source) = e.source() {
            error!("caused by {}", source.to_string());
        }
        // if let Some(backtrace) = e.backtrace() {
        //     error!("{}", backtrace.to_string())
        // }
    })?;
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

    let mauve = crate::curves::mauve(1.0);

    // parse textures from disk or from literal
    let mut textures_map: HashMap<String, _> = HashMap::new();

    match scene.textures.resolve() {
        Ok(textures_data) => {
            for (name, data) in textures_data {
                if let Some(parsed) = parse_texture_stack(data, &curves) {
                    textures_map.insert(name.clone(), parsed);
                } else {
                    warn!("failed to parse texture {}", name);
                }
            }
        }
        Err(e) => {
            error!("failed to resolve textures, error = {}", e.to_string());
            panic!()
        }
    }

    // parse enviroment
    let environment = parse_environment(scene.environment, &curves, &textures_map, &mauve);
    if environment.is_none() {
        error!("failed to parse environment");
        // TODO: change to return Err once an Errors enum exists
        panic!();
    }

    // parse materials from disk or from literal
    let mut materials_map: HashMap<String, _> = HashMap::new();
    match scene.materials.resolve() {
        Ok(materials_data) => {
            for (name, data) in materials_data {
                if let Some(parsed) = data.resolve(&curves, &textures_map) {
                    if materials_map.insert(name.clone(), parsed).is_none() {
                        info!("inserted new material {}", &name);
                    } else {
                        warn!("replaced material {}", &name);
                    }
                } else {
                    warn!("failed to parse material {}", name);
                }
            }
        }
        Err(e) => {
            error!("failed to parse materials, error = {}", e.to_string());
            panic!()
        }
    }

    // add mauve material to indicate errors.
    // completely black for reflection, emits mauve color
    let mauve = MaterialEnum::DiffuseLight(DiffuseLight::new(
        crate::curves::cie_e(0.0),
        mauve
            .clone()
            .to_cdf(math::spectral::EXTENDED_VISIBLE_RANGE, 20),
        math::Sidedness::Dual,
    ));

    // parse mediums from disk or from literal

    let mut mediums_map: HashMap<String, _> = HashMap::new();
    match scene.mediums.map(|e| e.resolve()) {
        Some(Ok(mediums_data)) => {
            for (name, data) in mediums_data {
                if let Some(parsed) = parse_medium(data, &curves) {
                    if mediums_map.insert(name.clone(), parsed).is_none() {
                        info!("inserted new medium {}", &name);
                    } else {
                        warn!("replaced medium {}", &name);
                    }
                } else {
                    warn!("failed to parse medium {}", name);
                }
            }
        }
        Some(Err(e)) => {
            error!("{}", e.to_string());
            panic!();
        }
        _ => {
            info!("no mediums found, continuing");
        }
    }

    // parse instances, and serialize all materials, mediums, and textures into vectors for storage in world.

    let mut materials: Vec<MaterialEnum> = Vec::new();
    let mut material_names_to_ids = HashMap::new();

    // 0th material is always the error material.
    // allows default material to be the error material when one is not found, specifically in Mesh and MeshTriangleRef
    let error_material_id = MaterialId::Light(0u16);
    materials.push(mauve);
    material_names_to_ids.insert(String::from("error"), error_material_id);
    for (name, material) in materials_map {
        let id = materials.len();
        let id = match &material {
            MaterialEnum::DiffuseLight(_) | MaterialEnum::SharpLight(_) => {
                MaterialId::Light(id as u16)
            }
            _ => MaterialId::Material(id as u16),
        };
        materials.push(material);
        info!("added material {} as {:?}", &name, id);
        assert!(material_names_to_ids.insert(name, id).is_none());
    }
    // put medium_names_to_ids here
    let mut mediums: Vec<MediumEnum> = Vec::new();
    for (_, medium) in mediums_map {
        mediums.push(medium);
    }

    let mut instances: Vec<Instance> = Vec::new();
    for instance in scene.instances {
        match instance.aggregate {
            AggregateData::MeshBundle(data) => {
                let meshes = load_obj_file(&data.filename, &material_names_to_ids);
                let transform: Option<Transform3> = instance.transform.clone().map(|e| e.into());
                for mut mesh in meshes {
                    let id = instances.len();
                    mesh.init();
                    let maybe_material_id = instance
                        .material_name
                        .clone()
                        .map(|s| material_names_to_ids[&s]);
                    info!(
                        "pushing instance of mesh from meshbundle, with material id {:?} from {:?}",
                        maybe_material_id,
                        instance.material_name.clone()
                    );
                    instances.push(Instance::new(
                        Aggregate::Mesh(mesh),
                        transform,
                        maybe_material_id,
                        id,
                    ));
                }
            }
            _ => {
                info!("parsing instance and primitive");
                let id = instances.len();
                let instance = parse_instance(instance, &material_names_to_ids, id);
                info!(
                    "done. pushing instance with material {:?} and instance id {}",
                    instance.material_id, instance.instance_id
                );
                instances.push(instance);
            }
        }
    }

    let world = World::new(
        instances,
        materials,
        mediums,
        environment.unwrap(),
        scene.env_sampling_probability.unwrap_or(0.5),
        AcceleratorType::BVH,
    );
    Ok(world)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parsing_materials_lib() {
        let materials = MaybeMaterialLib::Path(String::from("data/lib_materials.toml"))
            .resolve()
            .unwrap();

        for (name, data) in &materials {
            println!("{}", name);
            match data {
                MaterialData::DiffuseLight(_) => {}
                MaterialData::SharpLight(_) => {}
                MaterialData::Lambertian(_) => {}
                MaterialData::GGX(inner) => {
                    println!("{:?}", inner.eta);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_parsing_textures_lib() {
        let textures = MaybeTextureLib::Path(String::from("data/lib_textures.toml"))
            .resolve()
            .unwrap();

        for (name, _data) in &textures {
            println!("{}", name);
        }
    }

    #[test]
    fn test_parsing_curves_lib() {
        let curves = MaybeCurvesLib::Path(String::from("data/lib_curves.toml"))
            .resolve()
            .unwrap();

        for (name, _data) in &curves {
            println!("{}", name);
        }
    }

    #[test]
    fn test_parsing_complex_scene() {
        let world = construct_world(PathBuf::from("data/scenes/hdri_test_2.toml")).unwrap();
        println!("constructed world");
        for mat in &world.materials {
            let name = match mat {
                MaterialEnum::DiffuseLight(_) => "diffuse_light",
                MaterialEnum::Lambertian(_) => "lambertian",
                MaterialEnum::GGX(_) => "GGX",
                MaterialEnum::SharpLight(_) => "sharp_light",
            };
            println!("{}", name);
        }
    }

    #[test]
    fn test_world() {
        let _world = construct_world(PathBuf::from("data/scenes/test_prism.toml")).unwrap();
    }
}
