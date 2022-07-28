use crate::materials::*;
use crate::texture::TexStack;
use math::{spectral::BOUNDED_VISIBLE_RANGE, Curve, Sidedness};

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::CurveDataOrReference;

#[derive(Serialize, Deserialize, Clone)]
pub struct LambertianData {
    pub texture_id: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GGXData {
    pub alpha: f32,
    pub eta: CurveDataOrReference,
    pub eta_o: CurveDataOrReference,
    pub kappa: CurveDataOrReference,
    pub permeability: f32,
    pub outer_medium_id: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DiffuseLightData {
    pub color: CurveDataOrReference,
    pub sidedness: Sidedness,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SharpLightData {
    pub color: CurveDataOrReference,
    pub sidedness: Sidedness,
    pub sharpness: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PassthroughFilterData {
    pub color: CurveDataOrReference,
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

impl MaterialData {
    pub fn resolve(
        self,
        curve_mapping: &HashMap<String, Curve>,
        texture_mapping: &HashMap<String, TexStack>,
    ) -> Option<MaterialEnum> {
        match self {
            MaterialData::GGX(data) => {
                info!("parsing GGX");
                if let (Some(eta), Some(eta_o), Some(kappa)) = (
                    data.eta.resolve(curve_mapping),
                    data.eta_o.resolve(curve_mapping),
                    data.kappa.resolve(curve_mapping),
                ) {
                    Some(MaterialEnum::GGX(GGX::new(
                        data.alpha,
                        eta,
                        eta_o,
                        kappa,
                        data.permeability,
                        data.outer_medium_id.unwrap_or(0),
                    )))
                } else {
                    warn!("failed to resolve one of eta, eta_o, or kappa");
                    None
                }
            }
            MaterialData::Lambertian(data) => {
                info!("parsing Lambertian");
                let texture = texture_mapping
                    .get(&data.texture_id)
                    .expect("didn't find texture stack id for texture name")
                    .clone();
                Some(MaterialEnum::Lambertian(Lambertian::new(texture)))
            }
            MaterialData::SharpLight(data) => {
                info!("parsing SharpLight");
                // let color = parse_texture_stack(data.color);
                let color = data
                    .color
                    .resolve(curve_mapping)?
                    .to_cdf(BOUNDED_VISIBLE_RANGE, 100);
                Some(MaterialEnum::SharpLight(SharpLight::new(
                    color,
                    data.sharpness,
                    data.sidedness,
                )))
            }
            // MaterialData::PassthroughFilter(data) => {
            //     println!("parsing PassthroughFilter");
            //     // let color = parse_texture_stack(data.color);
            //     let color = Curve::from(data.color);
            //     MaterialEnum::PassthroughFilter(PassthroughFilter::new(
            //         color,
            //         data.outer_medium_id,
            //         data.inner_medium_id,
            //     ))
            // }
            MaterialData::DiffuseLight(data) => {
                info!("parsing DiffuseLight");
                // let color = parse_texture_stack(data.color);
                let color = data
                    .color
                    .resolve(curve_mapping)?
                    .to_cdf(BOUNDED_VISIBLE_RANGE, 100);
                Some(MaterialEnum::DiffuseLight(DiffuseLight::new(
                    color,
                    data.sidedness,
                )))
            }
            _ => panic!(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum MaybeMaterialLiteral {
    Literal(MaterialData),
    Named(String),
}
