use crate::materials::*;
use crate::parsing::curves::{parse_curve, CurveData};
use crate::texture::TexStack;
use math::Sidedness;

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct LambertianData {
    pub color: String,
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
    mapping: &HashMap<String, usize>,
    texture_stacks: &Vec<TexStack>,
) -> MaterialEnum {
    match data {
        MaterialData::GGX(data) => {
            println!("parsing GGX");
            let eta = parse_curve(data.eta);
            let kappa = parse_curve(data.kappa);
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
            let id = mapping
                .get(&data.color)
                .expect("didn't find texture stack id for texture name");
            MaterialEnum::Lambertian(Lambertian::new(texture_stacks[*id].clone()))
        }
        MaterialData::SharpLight(data) => {
            println!("parsing SharpLight");
            // let color = parse_texture_stack(data.color);
            let color = parse_curve(data.color).into();
            MaterialEnum::SharpLight(SharpLight::new(color, data.sharpness, data.sidedness))
        }
        MaterialData::PassthroughFilter(data) => {
            println!("parsing PassthroughFilter");
            // let color = parse_texture_stack(data.color);
            let color = parse_curve(data.color).into();
            MaterialEnum::PassthroughFilter(PassthroughFilter::new(
                color,
                data.outer_medium_id,
                data.inner_medium_id,
            ))
        }
        MaterialData::DiffuseLight(data) => {
            println!("parsing DiffuseLight");
            // let color = parse_texture_stack(data.color);
            let color = parse_curve(data.color).into();
            MaterialEnum::DiffuseLight(DiffuseLight::new(color, data.sidedness))
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NamedMaterial {
    pub data: MaterialData,
    pub name: String,
}
