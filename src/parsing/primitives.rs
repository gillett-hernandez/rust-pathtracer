use crate::prelude::*;

use crate::geometry::*;
use crate::parsing::Point3Data;

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct DiskData {
    pub radius: f32,
    pub origin: Point3Data,
    pub two_sided: bool,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct SphereData {
    pub radius: f32,
    pub origin: Point3Data,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct RectData {
    pub size: (f32, f32),
    pub normal: Axis,
    pub origin: Point3Data,
    pub two_sided: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct MeshRef {
    pub name: String,
    pub index: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum AggregateData {
    Disk(DiskData),
    Rect(RectData),
    Sphere(SphereData),
    Mesh(MeshRef),
}

impl AggregateData {
    pub fn parse_with(self, mesh_map: &HashMap<String, Mesh>) -> Aggregate {
        match self {
            AggregateData::Disk(data) => {
                assert!(data.radius > 0.0);
                info!("parsed disk data");
                Aggregate::Disk(Disk::new(data.radius, data.origin.into(), data.two_sided))
            }
            AggregateData::Rect(data) => {
                assert!(data.size.0 > 0.0 && data.size.1 > 0.0);
                info!("parsed rect data");
                Aggregate::AARect(AARect::new(
                    data.size,
                    data.origin.into(),
                    data.normal,
                    data.two_sided,
                ))
            }
            AggregateData::Sphere(data) => {
                assert!(data.radius > 0.0);
                info!("parsed sphere data");
                Aggregate::Sphere(Sphere::new(data.radius, data.origin.into()))
            }
            AggregateData::Mesh(data) => {
                let mesh = mesh_map.get(&data.name).expect("mesh map did not contain mesh. this should have been caught earlier, when analyzing materials");
                Aggregate::Mesh(mesh.clone())
            }
        }
    }
}
