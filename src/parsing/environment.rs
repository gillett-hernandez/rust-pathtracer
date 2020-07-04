use crate::parsing::curves::Curve;
use crate::parsing::Vec3Data;
use serde::{Deserialize, Serialize};

use crate::math::*;
use crate::parsing::curves::parse_curve;
use crate::world::EnvironmentMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct ConstantData {
    pub color: Curve,
    pub strength: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SunData {
    pub color: Curve,
    pub strength: f32,
    pub solid_angle: f32,
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
            solid_angle: data.solid_angle,
            sun_direction: Vec3::from(data.sun_direction),
        },
    }
}
