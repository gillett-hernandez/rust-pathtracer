use crate::geometry::*;
use crate::materials::MaterialId;
use crate::parsing::primitives::*;
use crate::parsing::Vec3Data;

use std::collections::HashMap;
// use std::env;
// use std::fs::File;
// use std::io::Read;
// use std::io::{self, BufWriter, Write};
// use std::path::Path;

use crate::prelude::PI;
use math::prelude::{Transform3, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone)]
pub struct AxisAngleData {
    pub axis: Vec3Data,
    pub angle: f32,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Transform3Data {
    pub scale: Option<Vec3Data>,            // scale
    pub rotate: Option<Vec<AxisAngleData>>, // axis angle
    pub translate: Option<Vec3Data>,        // translation
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct InstanceData {
    pub aggregate: AggregateData,
    pub transform: Option<Transform3Data>,
    pub material_name: Option<String>,
}

impl From<Transform3Data> for Transform3 {
    fn from(data: Transform3Data) -> Self {
        info!("parsing transform data");
        let maybe_scale = data.scale.map(|v| {
            info!("parsed scale");
            Transform3::from_scale(Vec3::from(v))
        });
        let maybe_rotate = if let Some(rotations) = data.rotate {
            let mut base = None;
            for rotation in rotations {
                let transform = Transform3::from_axis_angle(
                    Vec3::from(rotation.axis).normalized(),
                    PI * rotation.angle / 180.0,
                );
                info!("parsed rotate {:?} {}", rotation.axis, rotation.angle);
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
        let maybe_translate = data.translate.map(|v| {
            info!("parsed translate");
            Transform3::from_translation(Vec3::from(v))
        });
        Transform3::from_stack(maybe_scale, maybe_rotate, maybe_translate)
    }
}

pub fn parse_instance(
    instance_data: InstanceData,
    materials_mapping: &HashMap<String, MaterialId>,
    mesh_map: &HashMap<String, Mesh>,
    instance_id: InstanceId,
) -> Instance {
    let aggregate: Aggregate = instance_data.aggregate.parse_with(mesh_map);
    let transform: Option<Transform3> = instance_data
        .transform
        .map(|transform_data| transform_data.into());

    let material_id = match &instance_data.material_name {
        None => {
            warn!("material name on instance {} was none", instance_id);
            None
        }
        Some(name) => Some(match materials_mapping.get(name) {
            Some(id) => *id,
            None => {
                error!(
                    "material not found in mapping, instance {}, material name {}",
                    instance_id, name
                );
                *materials_mapping.get("error").unwrap()
            }
        }),
    };

    //  = match instance_data.material_name.clone() {
    //     Some(name) => materials_mapping.get(&name).cloned(),
    //     None => {
    //         error!();
    //         materials_mapping.get("error").cloned()
    //     },
    // };

    info!(
        "parsed instance, assigned material id {:?} from {:?}, and instance id {}",
        material_id,
        instance_data
            .material_name
            .unwrap_or_else(|| "None".to_string()),
        instance_id
    );
    Instance::new(aggregate, transform, material_id, instance_id)
}
