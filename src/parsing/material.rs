use crate::materials::*;
use crate::parsing::curves::CurveData;
use crate::texture::TexStack;
use math::{SPD, Sidedness};

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct LambertianData {
    pub texture_id: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GGXData {
    pub alpha: f32,
    pub eta_o: f32,
    pub eta: CurveData,
    pub kappa: CurveData,
    pub permeability: f32,
    pub outer_medium_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DiffuseLightData {
    pub color: CurveData,
    pub sidedness: Sidedness,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SharpLightData {
    pub color: CurveData,
    pub sidedness: Sidedness,
    pub sharpness: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PassthroughFilterData {
    pub color: CurveData,
    pub outer_medium_id: usize,
    pub inner_medium_id: usize,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum MaterialData {
    GGX(GGXData),
    Lambertian(LambertianData),
    PassthroughFilter(PassthroughFilterData),

    DiffuseLight(DiffuseLightData),
    SharpLight(SharpLightData),
}

pub fn parse_material(
    data: MaterialData,
    texture_mapping: &HashMap<String, usize>,
    textures: &Vec<TexStack>,
) -> MaterialEnum {
    match data {
        MaterialData::GGX(data) => {
            println!("parsing GGX");
            let eta = data.eta.into();
            let kappa = data.kappa.into();
            MaterialEnum::GGX(GGX::new(
                data.alpha,
                eta,
                data.eta_o,
                kappa,
                data.permeability,
                data.outer_medium_id.unwrap_or(0),
            ))
        }
        MaterialData::Lambertian(data) => {
            println!("parsing Lambertian");
            let id = texture_mapping
                .get(&data.texture_id)
                .expect("didn't find texture stack id for texture name");
            MaterialEnum::Lambertian(Lambertian::new(textures[*id].clone()))
        }
        MaterialData::SharpLight(data) => {
            println!("parsing SharpLight");
            // let color = parse_texture_stack(data.color);
            let color = SPD::from(data.color).into();
            MaterialEnum::SharpLight(SharpLight::new(color, data.sharpness, data.sidedness))
        }
        MaterialData::PassthroughFilter(data) => {
            println!("parsing PassthroughFilter");
            // let color = parse_texture_stack(data.color);
            let color = SPD::from(data.color).into();
            MaterialEnum::PassthroughFilter(PassthroughFilter::new(
                color,
                data.outer_medium_id,
                data.inner_medium_id,
            ))
        }
        MaterialData::DiffuseLight(data) => {
            println!("parsing DiffuseLight");
            // let color = parse_texture_stack(data.color);
            let color = SPD::from(data.color).into();
            MaterialEnum::DiffuseLight(DiffuseLight::new(color, data.sidedness))
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NamedMaterial {
    pub data: MaterialData,
    pub name: String,
}
