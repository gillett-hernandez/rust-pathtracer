use crate::materials::*;
use crate::parsing::curves::{parse_curve, Curve};
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
    pub eta: Curve,
    pub kappa: Curve,
    pub permeability: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DiffuseLightData {
    pub color: Curve,
    pub sidedness: Sidedness,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SharpLightData {
    pub color: Curve,
    pub sidedness: Sidedness,
    pub sharpness: f32,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum MaterialData {
    GGX(GGXData),
    Lambertian(LambertianData),

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
