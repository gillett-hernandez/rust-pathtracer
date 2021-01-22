use crate::parsing::curves::CurveData;
use crate::parsing::Vec3Data;
use serde::{Deserialize, Serialize};

use crate::math::*;
use crate::parsing::curves::parse_curve;
use crate::world::EnvironmentMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct ConstantData {
    pub color: CurveData,
    pub strength: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SunData {
    pub color: CurveData,
    pub strength: f32,
    pub angular_diameter: f32,
    pub sun_direction: Vec3Data,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum EnvironmentData {
    Constant(ConstantData),
    Sun(SunData),
}

pub fn parse_environment(env_data: EnvironmentData) -> EnvironmentMap {
    match env_data {
        EnvironmentData::Constant(data) => EnvironmentMap::Constant {
            color: parse_curve(data.color).into(),
            strength: data.strength,
        },
        EnvironmentData::Sun(data) => EnvironmentMap::Sun {
            color: parse_curve(data.color).into(),
            strength: data.strength,
            angular_diameter: data.angular_diameter,
            sun_direction: Vec3::from(data.sun_direction).normalized(),
        },
    }
}
