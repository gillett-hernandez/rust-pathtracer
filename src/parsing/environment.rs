use crate::prelude::*;

use std::collections::HashMap;

use math::curves::InterpolationMode;
use math::spectral::BOUNDED_VISIBLE_RANGE;
use serde::{Deserialize, Serialize};

use crate::texture::{Texture, Texture1};
use crate::world::{EnvironmentMap, ImportanceMap};

use super::curves::CurveData;
use super::instance::{AxisAngleData, Transform3Data};
use super::{CurveDataOrReference, Vec3Data};

#[derive(Serialize, Deserialize, Clone)]
pub struct ConstantData {
    pub color: CurveDataOrReference,
    pub strength: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SunData {
    pub color: CurveDataOrReference,
    pub strength: f32,
    pub angular_diameter: f32,
    pub sun_direction: Vec3Data,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ImportanceMapData {
    pub width: usize,
    pub height: usize,
    pub luminance_curve: Option<CurveData>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HDRIData {
    pub texture_name: String,
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
    curves: &HashMap<String, Curve>,
    textures: &HashMap<String, TexStack>,
    error_color: &Curve,
) -> Option<EnvironmentMap> {
    match env_data {
        EnvironmentData::Constant(data) => EnvironmentMap::Constant {
            color: data
                .color
                .resolve(curves)
                .unwrap_or_else(|| {
                    error!("failed to resolve curve, falling back to error color");
                    error_color.clone()
                })
                .to_cdf(BOUNDED_VISIBLE_RANGE, 100),
            strength: data.strength,
        },
        EnvironmentData::Sun(data) => EnvironmentMap::Sun {
            color: data
                .color
                .resolve(curves)
                .unwrap_or_else(|| {
                    error!("failed to resolve curve, falling back to error color");
                    error_color.clone()
                })
                .to_cdf(BOUNDED_VISIBLE_RANGE, 100),
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
            let texture = textures
                .get(&data.texture_name)
                .cloned()
                .unwrap_or_else(|| {
                    warn!("importance map texture not found, using mauve texture");
                    TexStack {
                        textures: vec![Texture::Texture1(Texture1 {
                            curve: error_color.to_cdf(BOUNDED_VISIBLE_RANGE, 100),
                            texture: Vec2D::new(1, 1, 1.0),
                            interpolation_mode: InterpolationMode::Linear,
                        })],
                    }
                });
            let importance_map = match data.importance_map {
                Some(data) => ImportanceMap::Unbaked {
                    horizontal_resolution: data.width,
                    vertical_resolution: data.height,
                    luminance_curve: data
                        .luminance_curve
                        .map(|e| e.into())
                        .unwrap_or_else(Curve::y_bar),
                },
                None => ImportanceMap::Empty,
            };

            EnvironmentMap::HDR {
                texture,
                rotation,
                importance_map,
                strength: data.strength,
            }
        }
    }
    .into()
}
