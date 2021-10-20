use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::math::*;
use crate::texture::TexStack;
use crate::world::EnvironmentMap;

use super::curves::{parse_curve, CurveData};
use super::instance::{AxisAngleData, Transform3Data};
use super::Vec3Data;

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
pub struct HDRIData {
    pub texture_id: String,
    pub strength: f32,
    pub rotation: Vec<AxisAngleData>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum EnvironmentData {
    Constant(ConstantData),
    Sun(SunData),
    HDRI(HDRIData),
}

pub fn parse_environment(
    env_data: EnvironmentData,
    texture_mapping: &HashMap<String, usize>,
    textures: &Vec<TexStack>,
) -> EnvironmentMap {
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
        EnvironmentData::HDRI(data) => EnvironmentMap::HDRi {
            texture: textures[*texture_mapping
                .get(&data.texture_id)
                .expect("requested texture id was not in texture mapping")]
            .clone(),
            rotation: Transform3Data {
                scale: None,
                rotate: Some(data.rotation),
                translate: None,
            }
            .into(),
            strength: data.strength,
        },
    }
}
