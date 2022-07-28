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
    let data = tobj::load_obj(
        filename,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    );
    info!("opening file at {}", filename);
    assert!(data.is_ok(), "{:?}", data);

    let (models, materials) = data.expect("Failed to load OBJ file");
    let materials = materials.expect("Failed to load MTL file");
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
    let data = tobj::load_obj(
        filename,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    );
    info!("opening file at {}", filename);
    assert!(data.is_ok(), "{:?}", data);
    let (models, materials) = data.expect("Failed to load OBJ file");
    let materials = materials.expect("Failed to load MTL file");
    parse_obj_mesh(&models, &materials, obj_id, material_mapping)
}

pub fn parse_obj_mesh(
    models: &Vec<tobj::Model>,
    materials: &Vec<tobj::Material>,
    obj_num: usize,
    material_mapping: &HashMap<String, MaterialId>,
) -> Mesh {
    info!("# of models: {}", models.len());
    info!("# of materials: {}", materials.len());
    let mut points = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut num_faces = 0;
    // for (i, m) in models.iter().enumerate() {
    let model = &models[obj_num];
    let mesh = &model.mesh;
    info!("model[{}].name = \'{}\'", obj_num, model.name);
    info!(
        "model[{}].mesh.material_id = {:?}",
        obj_num, mesh.material_id
    );

    for mat in materials.iter() {
        info!("{}", mat.name);
    }

    // let estimated_num_faces = if mesh.face_arities.len() > 0 {
    //     mesh.face_arities.len()
    // } else {
    //     mesh.indices.len() / 3
    // };

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

    let mut next_face = 0;
    // mesh.
    if mesh.face_arities.is_empty() {
        // all faces are triangles
        info!(
            "no face arities, thus all are triangles. {} faces",
            mesh.indices.len() / 3
        );
        indices.extend(mesh.indices.iter().map(|e| *e as usize));
        num_faces += mesh.indices.len() / 3;
    } else {
        // handle non-triangle faces.
        for f in 0..mesh.face_arities.len() {
            let end = next_face + mesh.face_arities[f] as usize;
            let face_indices: Vec<_> = mesh.indices[next_face..end].iter().collect();
            indices.push(*face_indices[0] as usize);
            indices.push(*face_indices[1] as usize);
            indices.push(*face_indices[2] as usize);
            // println!("    face[{}] = {:?}", f, face_indices);
            next_face = end;
            num_faces += 1;
        }
    }
    let mat_ids = vec![mesh.material_id.unwrap_or(0); num_faces]
        .iter()
        .map(|idx| mesh_materials[*idx])
        .collect();
    info!("Size of model[{}].num_face_indices: {}", obj_num, num_faces);

    // Normals and texture coordinates are also loaded, but not printed in this example
    info!("model[{}].vertices: {}", obj_num, mesh.positions.len() / 3);
    assert!(mesh.positions.len() % 3 == 0);
    for v in 0..mesh.positions.len() / 3 {
        points.push(Point3::new(
            mesh.positions[3 * v],
            mesh.positions[3 * v + 1],
            mesh.positions[3 * v + 2],
        ))
    }
    if mesh.normals.len() > 0 {
        info!("parsing normals");

        for n in 0..mesh.normals.len() / 3 {
            normals.push(Vec3::new(
                mesh.normals[3 * n],
                mesh.normals[3 * n + 1],
                mesh.normals[3 * n + 2],
            ))
        }
        info!("parsed normals, len = {}", normals.len());
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
        match self {
            AggregateData::Disk(data) => {
                info!("parsed disk data");
                Aggregate::Disk(Disk::new(data.radius, data.origin.into(), data.two_sided))
            }
            AggregateData::Rect(data) => {
                info!("parsed rect data");
                Aggregate::AARect(AARect::new(
                    data.size,
                    data.origin.into(),
                    data.normal,
                    data.two_sided,
                ))
            }
            AggregateData::Sphere(data) => {
                info!("parsed sphere data");
                Aggregate::Sphere(Sphere::new(data.radius, data.origin.into()))
            }
            AggregateData::Mesh(data) => {
                info!("parsed Mesh data");
                let filename = data.filename;
                let mut mesh =
                    parse_specific_obj_mesh(&filename, data.mesh_index, material_mapping);
                info!("parsed mesh, initializing mesh and mesh bvh");
                mesh.init();
                info!("initialized mesh and mesh bvh");
                Aggregate::Mesh(mesh)
            }
            // parse_with does not handle MeshBundle
            _ => panic!(),
        }
    }
}
