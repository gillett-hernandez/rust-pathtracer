extern crate num_cpus;
extern crate serde;

use crate::parsing::Vec3Data;

use crate::geometry::*;
use crate::materials::MaterialId;
use crate::math::*;
use crate::parsing::primitives::*;

use std::collections::HashMap;

// use std::env;
// use std::fs::File;
// use std::io::Read;
// use std::io::{self, BufWriter, Write};
// use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct Transform3Data {
    pub scale: Option<Vec3Data>,         // scale
    pub rotate: Option<(Vec3Data, f32)>, // axis angle
    pub translate: Option<Vec3Data>,     // translation
}

#[derive(Serialize, Deserialize, Clone)]
pub struct InstanceData {
    pub aggregate: AggregateData,
    pub transform: Option<Transform3Data>,
    pub material_identifier: String,
}

pub fn parse_instance(
    instance_data: InstanceData,
    materials_mapping: &HashMap<String, MaterialId>,
    instance_id: usize,
) -> Instance {
    let aggregate: Aggregate = instance_data.aggregate.into();
    let transform = instance_data.transform.map(|transform_data| {
        let maybe_scale = transform_data
            .scale
            .map(|v| Transform3::from_scale(Vec3::from(v)));
        let maybe_rotate = transform_data
            .rotate
            .map(|v| Transform3::from_axis_angle(Vec3::from(v.0), v.1));
        let maybe_translate = transform_data
            .translate
            .map(|v| Transform3::from_translation(Vec3::from(v)));
        Transform3::from_stack(maybe_scale, maybe_rotate, maybe_translate)
    });
    let material_id = *materials_mapping
        .get(&instance_data.material_identifier)
        .expect("material mapping did not contain material name");
    println!(
        "parsed instance, assigned material id {:?} and instance id {}",
        material_id, instance_id
    );
    Instance::new(aggregate, transform, material_id, instance_id)
}
