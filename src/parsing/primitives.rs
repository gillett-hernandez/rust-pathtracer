extern crate num_cpus;
extern crate serde;

use crate::geometry::*;
use crate::materials::MaterialId;
use crate::math::*;
use crate::parsing::Point3Data;

use serde::{Deserialize, Serialize};
use tobj;

use std::collections::HashMap;

pub fn load_obj_file(filename: &str, material_mapping: &HashMap<String, MaterialId>) -> Vec<Mesh> {
    let data = tobj::load_obj(filename, true);
    println!("opening file at {}", filename);
    assert!(data.is_ok(), "{:?}", data);
    let (models, materials) = data.unwrap();
    let mut meshes = Vec::new();
    for i in 0..models.len() {
        meshes.push(parse_obj_mesh(&models, &materials, i, material_mapping));
    }
    meshes
}

pub fn parse_specific_obj_mesh(
    filename: &str,
    obj_id: usize,
    material_mapping: &HashMap<String, MaterialId>,
) -> Mesh {
    let data = tobj::load_obj(filename, true);
    println!("opening file at {}", filename);
    assert!(data.is_ok(), "{:?}", data);
    let (models, materials) = data.unwrap();
    parse_obj_mesh(&models, &materials, obj_id, material_mapping)
}

pub fn parse_obj_mesh(
    models: &Vec<tobj::Model>,
    materials: &Vec<tobj::Material>,
    obj_num: usize,
    material_mapping: &HashMap<String, MaterialId>,
) -> Mesh {
    println!("# of models: {}", models.len());
    println!("# of materials: {}", materials.len());
    let mut points = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut num_faces = 0;
    // for (i, m) in models.iter().enumerate() {
    let model = &models[obj_num];
    let mesh = &model.mesh;
    println!("model[{}].name = \'{}\'", obj_num, model.name);
    println!(
        "model[{}].mesh.material_id = {:?}",
        obj_num, mesh.material_id
    );

    for mat in materials.iter() {
        println!("{}", mat.name);
    }

    let mesh_materials: Vec<MaterialId> = materials
        .iter()
        .map(|v| {
            if let Some(mat_ref) = material_mapping.get(&v.name) {
                *mat_ref
            } else {
                0.into()
            }
        })
        .collect();

    let mat_ids = vec![mesh.material_id.unwrap_or(0); mesh.num_face_indices.len()]
        .iter()
        .map(|idx| mesh_materials[*idx])
        .collect();
    println!(
        "Size of model[{}].num_face_indices: {}",
        obj_num,
        mesh.num_face_indices.len()
    );
    let mut next_face = 0;
    for f in 0..mesh.num_face_indices.len() {
        let end = next_face + mesh.num_face_indices[f] as usize;
        let face_indices: Vec<_> = mesh.indices[next_face..end].iter().collect();
        indices.push(*face_indices[0] as usize);
        indices.push(*face_indices[1] as usize);
        indices.push(*face_indices[2] as usize);
        // println!("    face[{}] = {:?}", f, face_indices);
        next_face = end;
        num_faces += 1;
    }

    // Normals and texture coordinates are also loaded, but not printed in this example
    println!("model[{}].vertices: {}", obj_num, mesh.positions.len() / 3);
    assert!(mesh.positions.len() % 3 == 0);
    for v in 0..mesh.positions.len() / 3 {
        points.push(Point3::new(
            mesh.positions[3 * v],
            mesh.positions[3 * v + 1],
            mesh.positions[3 * v + 2],
        ))
    }
    if mesh.normals.len() > 0 {
        println!("parsing normals");
    }
    for n in 0..mesh.normals.len() / 3 {
        normals.push(Vec3::new(
            mesh.normals[3 * n],
            mesh.normals[3 * n + 1],
            mesh.normals[3 * n + 2],
        ))
    }

    Mesh::new(num_faces, indices, points, normals, mat_ids)
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct DiskData {
    pub radius: f32,
    pub origin: Point3Data,
    pub two_sided: bool,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct SphereData {
    pub radius: f32,
    pub origin: Point3Data,
}

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct RectData {
    pub size: (f32, f32),
    pub normal: Axis,
    pub origin: Point3Data,
    pub two_sided: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MeshData {
    pub filename: String,
    pub mesh_index: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MeshBundleData {
    pub filename: String,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AggregateData {
    Disk(DiskData),
    Rect(RectData),
    Sphere(SphereData),
    Mesh(MeshData),
    MeshBundle(MeshBundleData),
}

impl AggregateData {
    pub fn parse_with(self, material_mapping: &HashMap<String, MaterialId>) -> Aggregate {
        // put mesh parsing here?
        match self {
            AggregateData::Disk(data) => {
                println!("parsed disk data");
                Aggregate::Disk(Disk::new(data.radius, data.origin.into(), data.two_sided))
            }
            AggregateData::Rect(data) => {
                println!("parsed rect data");
                Aggregate::AARect(AARect::new(
                    data.size,
                    data.origin.into(),
                    data.normal,
                    data.two_sided,
                ))
            }
            AggregateData::Sphere(data) => {
                println!("parsed sphere data");
                Aggregate::Sphere(Sphere::new(data.radius, data.origin.into()))
            }
            AggregateData::Mesh(data) => {
                println!("parsed Mesh data");
                let filename = data.filename;
                let mut mesh =
                    parse_specific_obj_mesh(&filename, data.mesh_index, material_mapping);
                mesh.init();
                Aggregate::Mesh(mesh)
            }
            // parse_with does not handle MeshBundle
            _ => panic!(),
        }
    }
}
