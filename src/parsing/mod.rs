use crate::mediums::MediumEnum;
use crate::prelude::*;

use crate::accelerator::AcceleratorType;
use crate::geometry::*;

use crate::materials::*;
use crate::world::World;

use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

pub mod cameras;
pub mod config;
pub mod curves;
mod environment;
mod instance;
mod material;
mod medium;
mod meshes;
pub mod primitives;
pub mod texture;
pub mod tonemap;

pub use cameras::*;
use config::*;
use environment::{parse_environment, EnvironmentData};
use instance::*;
use material::*;
use math::Transform3;
use medium::*;
use serde::de::DeserializeOwned;
use texture::*;
pub use tonemap::parse_tonemapper;

use curves::{CurveData, CurveDataOrReference};

use serde::{Deserialize, Serialize};
use toml;

use self::meshes::{load_obj_file, parse_specific_obj_mesh, MeshData};

use self::primitives::{AggregateData, MeshRef};

pub type Vec3Data = [f32; 3];
pub type Point3Data = [f32; 3];

macro_rules! unpack_or_LaP {
    ($e:expr, $($message: tt)*) => {
        match $e {
            Ok(inner) => inner,
            Err(err) => {
                error!($($message)*);
                error!("{:?}", err.to_string());
                // if let Some(backtrace) = err.backtrace() {
                //     error!("{:?}", backtrace);
                // }
                panic!()
            }
        }
    };
}

macro_rules! generate_maybe_libs {
    ($($e:ident),+) => {
        $(

            paste!{
                #[derive(Serialize, Deserialize, Clone)]
                #[serde(untagged)]
                pub enum [<Maybe $e Lib>] {
                    Literal(HashMap<String, [<$e Data>]>),
                    Path(String),
                }

                impl [<Maybe $e Lib>] {
                    pub fn resolve(self) -> HashMap<String, [<$e Data>]> {
                        match self {
                            Self::Literal(data) => data,
                            Self::Path(path) =>
                                unpack_or_LaP!(load_arbitrary(PathBuf::from(path.clone())), "failed to resolve path {}", path)

                        }
                    }
                }
            }

        )+
    };
}

generate_maybe_libs! {Curve, TextureStack, Material, Mesh, Medium}

// TODO: add a world storage for meshes so that instances don't clone them.
// maybe use Arcs around meshes? or let Mesh hold an actual reference to the actual Mesh data?
// verify how much arcs mess with performance compared to unsafe raw ptr access

#[derive(Serialize, Deserialize, Clone)]
pub struct SceneData {
    pub env_sampling_probability: Option<f32>,
    pub environment: EnvironmentData,
    pub curves: MaybeCurveLib,
    pub textures: MaybeTextureStackLib,
    pub materials: MaybeMaterialLib,
    pub mediums: Option<MaybeMediumLib>,
    pub meshes: MaybeMeshLib,
    pub instances: Vec<InstanceData>,
}

pub fn load_arbitrary<T: AsRef<Path>, O>(filepath: T) -> Result<O, Box<dyn Error>>
where
    O: DeserializeOwned,
{
    info!("loading file at {}", filepath.as_ref().to_string_lossy());
    let mut input = String::new();

    File::open(filepath.as_ref())
        .and_then(|mut f| f.read_to_string(&mut input))
        .expect("failed to load file");

    let data: O = toml::from_str(&input)?;
    Ok(data)
}

pub fn load_scene<T: AsRef<Path>>(filepath: T) -> Result<SceneData, Box<dyn Error>> {
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    let result = File::open(filepath.as_ref()).and_then(|mut f| f.read_to_string(&mut input));

    info!("loading file, {}", filepath.as_ref().to_string_lossy());
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
    })?;
    Ok(scene)
}

pub fn construct_world<P: AsRef<Path>>(scene_file: P) -> Result<World, Box<dyn Error>> {
    // layout of this function:
    // parse scene data from file
    // scan environment data for used textures/curves
    // scan instances for used meshes
    // scan meshes for used materials
    // scan materials for used textures
    // scan textures for used curves
    // parse and load used curves
    // parse and load used textures (expensive step, part of the reason the scanning pass exists)
    // parse and load used materials
    // parse and load used meshes
    // parse and load used instances

    // parse scene from disk

    let scene = unpack_or_LaP!(load_scene(scene_file), "failed to resolve scene file");

    // collect information from instances, materials, and env map data to determine what textures and materials actually need to be parsed and what can be discarded.

    let mut used_mediums = HashSet::new();
    let mut used_materials = HashSet::new();
    let mut used_textures = HashSet::new();
    let mut used_curves = HashSet::new();
    let mut used_meshes = HashSet::new();

    // scan env
    match &scene.environment {
        EnvironmentData::Constant(environment::ConstantData {
            color: CurveDataOrReference::Reference(name),
            ..
        }) => {
            used_curves.insert(name.clone());
        }
        EnvironmentData::Sun(environment::SunData {
            color: CurveDataOrReference::Reference(name),
            ..
        }) => {
            used_curves.insert(name.clone());
        }
        EnvironmentData::HDRI(environment::HDRIData { texture_name, .. }) => {
            used_textures.insert(texture_name.clone());
        }
        _ => {}
    }

    // scan instances
    for instance in &scene.instances {
        if let Some(ref name) = instance.material_name {
            used_materials.insert(name.clone());
        }
        if let AggregateData::Mesh(MeshRef { name, .. }) = &instance.aggregate {
            used_meshes.insert(name.clone());
        }
        // other aggregate types don't have associated material data
    }

    // scan meshes for used materials.
    // since parsing and loading the actual meshes is required here, keep the mesh data stored somewhere so we can use it later.
    let mesh_data = scene.meshes.resolve();

    let mut mesh_mapping: HashMap<String, Mesh> = HashMap::new();
    let mut mesh_material_mapping: HashMap<String, Vec<String>> = HashMap::new();
    // key 0 is mesh name,
    // value is vec of material names, matching the order that they are defined in the .mtl file
    // and thus matching the indices used in the actual mesh data.

    for (name, mesh_data) in mesh_data.iter().filter(|(k, _)| used_meshes.contains(*k)) {
        // parse mesh data, and add materials to used materials.
        assert!(
            !name.contains(';'),
            "semicolon (;) disallowed in mesh names"
        );
        let mut local_material_map = HashMap::new();
        match mesh_data.mesh_index {
            Some(index) => {
                let mesh = parse_specific_obj_mesh(
                    mesh_data.filename.as_str(),
                    index,
                    &mut local_material_map,
                );
                mesh_mapping.insert(name.clone(), mesh);
            }
            None => {
                let meshes = load_obj_file(mesh_data.filename.as_str(), &mut local_material_map);
                for (index, mesh) in meshes.into_iter().enumerate() {
                    let mut new_name = name.clone();
                    new_name.push(';');
                    new_name.push_str(&index.to_string());
                    mesh_mapping.insert(new_name, mesh);
                }
            }
        }
        for mat in local_material_map.keys() {
            used_materials.insert(mat.clone());
        }
        let length = local_material_map.len();
        // let mut vec = Vec::with_capacity(length + 1);
        let mut vec = vec![String::from(""); length];

        // vec.fill(String::new());
        local_material_map
            .into_iter()
            .for_each(|(name, index)| vec[index] = name);
        if length == 0 {
            vec.push(String::from("error"));
        }
        mesh_material_mapping.insert(name.clone(), vec);
    }

    // scan materials for used textures and curves
    let materials_data = scene.materials.resolve();

    for (_, material) in materials_data
        .iter()
        .filter(|(name, _)| used_materials.contains(*name))
    {
        match material {
            MaterialData::GGX(GGXData {
                eta,
                eta_o,
                kappa,
                outer_medium_id,
                ..
            }) => {
                for curve in &[eta, eta_o, kappa] {
                    if let Some(name) = curve.get_name() {
                        used_curves.insert(name.to_string());
                    }
                }
                if let Some(name) = outer_medium_id {
                    used_mediums.insert(name.clone());
                }
            }
            MaterialData::Lambertian(LambertianData { texture_id: name }) => {
                used_textures.insert(name.clone());
            }
            MaterialData::PassthroughFilter(PassthroughFilterData {
                color,
                inner_medium_id,
                outer_medium_id,
            }) => {
                if let Some(name) = color.get_name() {
                    used_curves.insert(name.to_string());
                }

                used_mediums.insert(inner_medium_id.clone());
                used_mediums.insert(outer_medium_id.clone());
            }
            MaterialData::SharpLight(SharpLightData {
                bounce_color,
                emit_color,
                ..
            })
            | MaterialData::DiffuseLight(DiffuseLightData {
                bounce_color,
                emit_color,
                ..
            }) => {
                if let Some(name) = bounce_color.get_name() {
                    used_curves.insert(name.to_string());
                }
                if let Some(name) = emit_color.get_name() {
                    used_curves.insert(name.to_string());
                }
            }
        }
    }

    // scan textures for used curves
    let textures_data = scene.textures.resolve();

    for (_, texture) in textures_data
        .iter()
        .filter(|(name, _)| used_textures.contains(*name))
    {
        for layer in &texture.0 {
            match layer {
                TextureData::Texture1 { curve, .. } => {
                    if let Some(name) = curve.get_name() {
                        used_curves.insert(name.to_string());
                    }
                }
                TextureData::Texture4 { curves, .. }
                | TextureData::HDR { curves, .. }
                | TextureData::EXR { curves, .. } => {
                    curves.iter().for_each(|curve| {
                        if let Some(name) = curve.get_name() {
                            used_curves.insert(name.to_string());
                        }
                    });
                }
                TextureData::SRGB { .. } => {} // uses in-code parsing of sRGB basis functions
            }
        }
    }

    // parse curves from disk or from literal,
    let curves_data = scene.curves.resolve();
    let curves = curves_data
        .into_iter()
        .filter(|(name, _)| used_curves.contains(name))
        .map(|(k, v)| (k, v.into()))
        .collect();

    let mauve = crate::curves::mauve(1.0);

    // parse textures from disk or from literal
    let mut textures_map: HashMap<String, TexStack> = HashMap::new();

    for (name, data) in textures_data
        .iter()
        .filter(|(name, _)| used_textures.contains(*name))
    {
        if let Some(parsed) = parse_texture_stack(data.clone(), &curves) {
            textures_map.insert(name.clone(), parsed);
        } else {
            warn!("failed to parse texture {}", name);
        }
    }

    // parse enviroment
    let environment = parse_environment(scene.environment, &curves, &textures_map, &mauve);
    if environment.is_none() {
        error!("failed to parse environment");
        // TODO: change to return Err once an Errors enum exists
        panic!();
    }

    // parse mediums from disk or from literal
    let mut mediums_map: HashMap<String, _> = HashMap::new();
    match scene.mediums.map(|e| e.resolve()) {
        Some(mediums_data) => {
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
        None => {
            info!("no mediums data, continuing")
        }
    }

    // dbg!(material_names_to_ids);
    // put medium_names_to_ids here
    let mut mediums_to_ids = HashMap::new();
    let mut mediums: Vec<MediumEnum> = Vec::new();
    for (name, medium) in mediums_map {
        let id = mediums.len();
        mediums.push(medium);
        mediums_to_ids.insert(name, id);
    }

    // parse materials from disk or from literal
    let mut materials_map: HashMap<String, _> = HashMap::new();

    for (name, data) in materials_data
        .into_iter()
        .filter(|(name, _)| used_materials.contains(name))
    {
        if let Some(parsed) = data.resolve(&curves, &textures_map, &mediums_to_ids) {
            if materials_map.insert(name.clone(), parsed).is_none() {
                info!("inserted new material {}", &name);
            } else {
                warn!("replaced material {}", &name);
            }
        } else {
            warn!("failed to parse material {}", name);
        }
    }

    // add mauve material to indicate errors.
    // completely black for reflection, emits mauve color
    let mauve = MaterialEnum::DiffuseLight(DiffuseLight::new(
        crate::curves::cie_e(0.0),
        mauve.to_cdf(math::spectral::EXTENDED_VISIBLE_RANGE, 20),
        math::Sidedness::Dual,
    ));

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

    // now that materials have been fully parsed and loaded, we can now go into each mesh and convert the dummy material ids
    // to their real material ids

    let mut mesh_mapping = mesh_mapping
        .into_iter()
        .map(|(name, mut mesh)| {
            info!("remapping material ids for mesh {}", &name);
            let prefix = name.split(';').next().unwrap();
            let materials_for_mesh = &mesh_material_mapping[prefix];

            // unwrap and transform material ids
            let mut material_ids = Arc::<_>::try_unwrap(mesh.material_ids)
                .expect("another Arc pointing to this mesh exists, and this unwrap failed");
            for mat_id in material_ids.iter_mut() {
                let index: usize = (*mat_id).into();
                let material_name = &materials_for_mesh[index];

                *mat_id = match material_names_to_ids
                    .get(material_name)
                    .cloned() {
                        Some(mat_id) => mat_id,
                        None => {
                            warn!("setting material ids to 0 since {} was not found in the materials library", material_name);
                            0.into()
                        }
                    }
            }
            mesh.material_ids = Arc::new(material_ids);
            info!("remapped successfully");
            (name, mesh)
        })
        .collect::<HashMap<_, _>>();

    // construct instances, initializing meshes along the way.
    // TODO: this would be included in any potential refactor that adds a world-wide storage location for Meshes.
    let mut instances: Vec<Instance> = Vec::new();
    for instance in scene.instances {
        match instance.aggregate {
            AggregateData::Mesh(MeshRef { name, index: None }) => {
                // bundle

                let transform: Option<Transform3> = instance.transform.clone().map(|e| e.into());
                for (mesh_name, mesh) in mesh_mapping.iter_mut() {
                    if !mesh_name.starts_with(&name) {
                        continue;
                    }
                    let id = instances.len();
                    mesh.init();
                    let maybe_material_id = instance
                        .material_name
                        .clone()
                        .map(|s| material_names_to_ids[&s]);
                    if maybe_material_id.is_some() {
                        info!(
                            "pushing instance of mesh from meshbundle, with material overridden to material id {:?} from {:?}",
                            maybe_material_id,
                            instance.material_name.clone()
                        );
                    } else {
                        info!("pusing instance of mesh from meshbundle, no material override");
                    }
                    instances.push(Instance::new(
                        Aggregate::Mesh(mesh.clone()),
                        transform,
                        maybe_material_id,
                        id,
                    ));
                }
            }
            _ => {
                info!("parsing instance and primitive");
                let id = instances.len();
                let instance = parse_instance(instance, &material_names_to_ids, &mesh_mapping, id);
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

pub fn get_settings(filepath: String) -> Result<TOMLConfig, toml::de::Error> {
    // will return None in the case that it can't read the settings file for whatever reason.
    // TODO: convert this to return Result<Settings, UnionOfErrors>
    let mut input = String::new();
    File::open(&filepath)
        .and_then(|mut f| f.read_to_string(&mut input))
        .unwrap();
    let num_cpus = num_cpus::get();
    let mut settings: TOMLConfig = toml::from_str(&input)?;
    for render_settings in settings.render_settings.iter_mut() {
        render_settings.threads = match render_settings.threads {
            Some(expr) => Some(expr),
            None => Some(num_cpus as u16),
        };
    }
    Ok(settings)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parsing_materials_lib() {
        let materials = MaybeMaterialLib::Path(String::from("data/lib_materials.toml")).resolve();

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
        let textures = MaybeTextureStackLib::Path(String::from("data/lib_textures.toml")).resolve();

        for (name, _data) in &textures {
            println!("{}", name);
        }
    }

    #[test]
    fn test_parsing_curves_lib() {
        let curves = MaybeCurveLib::Path(String::from("data/lib_curves.toml")).resolve();

        for (name, _data) in &curves {
            println!("{}", name);
        }
    }

    #[test]
    fn test_parsing_meshes_lib() {
        let meshes = MaybeMeshLib::Path(String::from("data/lib_meshes.toml")).resolve();

        for (name, _data) in &meshes {
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
                MaterialEnum::PassthroughFilter(_) => "passthrough filter",
            };
            println!("{}", name);
        }
    }

    #[test]
    fn test_world() {
        let _world = construct_world(PathBuf::from("data/scenes/test_prism.toml")).unwrap();
    }

    #[test]
    fn test_parsing_config() {
        let settings: TOMLConfig = match get_settings("data/config.toml".to_string()) {
            Ok(expr) => expr,
            Err(v) => {
                println!("{:?}", "couldn't read config.toml");
                println!("{:?}", v);
                return;
            }
        };
        for config in &settings.render_settings {
            assert!(config.filename != None);
            assert!(config.threads.unwrap() > 0)
        }
    }
}
