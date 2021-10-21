use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::math::*;
use crate::texture::TexStack;
use crate::world::{EnvironmentMap, ImportanceMap};

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
pub struct ImportanceMapData {
    pub width: usize,
    pub height: usize,
    pub luminance_curve: Option<SPD>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HDRIData {
    pub texture_id: String,
    pub strength: f32,
    pub rotation: Option<Vec<AxisAngleData>>,
    pub importance_map: Option<ImportanceMapData>,
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
        EnvironmentData::HDRI(data) => {
            let rotation = Transform3Data {
                scale: None,
                rotate: data.rotation,
                translate: None,
            }
            .into();
            let texture = textures[*texture_mapping
                .get(&data.texture_id)
                .expect("requested texture id was not in texture mapping")]
            .clone();
            let importance_map = if let Some(importance_map_data) = data.importance_map {
                Some(ImportanceMap::new(
                    texture,
                    importance_map_data.height,
                    importance_map_data.width,
                    importance_map_data.luminance_curve.unwrap_or_else(|| {
                        SPD::y_bar();
                    }),
                ))
            } else {
                None
            };
            EnvironmentMap::HDRi {
                texture,
                rotation,
                importance_map,
                strength: data.strength,
            }
        }
    }
}
