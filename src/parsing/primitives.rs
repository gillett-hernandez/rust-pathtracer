extern crate num_cpus;
extern crate serde;

use crate::geometry::*;
use crate::math::Axis;
use crate::parsing::Point3Data;

use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize, Copy, Clone)]
#[serde(tag = "type")]
pub enum AggregateData {
    Disk(DiskData),
    Rect(RectData),
    Sphere(SphereData),
}

impl From<AggregateData> for Aggregate {
    fn from(aggregate_data: AggregateData) -> Self {
        // put mesh parsing here?
        match aggregate_data {
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
        }
    }
}
