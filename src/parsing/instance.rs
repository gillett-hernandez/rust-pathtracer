extern crate num_cpus;
extern crate serde;

use crate::parsing::Vec3Data;

use crate::geometry::*;
use crate::materials::MaterialId;
use crate::parsing::primitives::*;

use std::collections::HashMap;

// use std::env;
// use std::fs::File;
// use std::io::Read;
// use std::io::{self, BufWriter, Write};
// use std::path::Path;

use serde::{Deserialize, Serialize};
#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct AxisAngleData {
    pub axis: Vec3Data,
    pub angle: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Transform3Data {
    pub scale: Option<Vec3Data>,            // scale
    pub rotate: Option<Vec<AxisAngleData>>, // axis angle
    pub translate: Option<Vec3Data>,        // translation
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InstanceData {
    pub aggregate: AggregateData,
    pub transform: Option<Transform3Data>,
    pub material_identifier: Option<String>,
}

pub fn parse_instance(
    instance_data: InstanceData,
    materials_mapping: &HashMap<String, MaterialId>,
    instance_id: usize,
) -> Instance {
    let aggregate: Aggregate = instance_data.aggregate.parse_with(materials_mapping);
    let transform = instance_data
        .transform
        .map(|transform_data| transform_data.into());

    let material_id = instance_data.material_identifier.clone().map(|v| {
        *materials_mapping
            .get(&v)
            .expect("material mapping did not contain material name")
    });
    println!(
        "parsed instance, assigned material id {:?} from {:?}, and instance id {}",
        material_id,
        instance_data
            .material_identifier
            .unwrap_or("None".to_string()),
        instance_id
    );
    Instance::new(aggregate, transform, material_id, instance_id)
}
