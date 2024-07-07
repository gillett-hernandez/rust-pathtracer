use std::collections::HashMap;

use crate::mediums::*;
use math::prelude::Curve;

use serde::{Deserialize, Serialize};

use super::CurveDataOrReference;

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct HGMediumData {
    pub g: CurveDataOrReference,
    pub sigma_s: CurveDataOrReference,
    pub sigma_a: CurveDataOrReference,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RayleighData {
    pub ior: CurveDataOrReference,
    pub corrective_factor: f32,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
#[serde(tag = "type")]
pub enum MediumData {
    HG(HGMediumData),
    Rayleigh(RayleighData),
}

pub fn parse_medium(data: MediumData, curves: &HashMap<String, Curve>) -> Option<MediumEnum> {
    match data {
        MediumData::HG(data) => {
            info!("parsing HG");
            let g = data.g.resolve(curves)?;
            let sigma_s = data.sigma_s.resolve(curves)?;
            let sigma_a = data.sigma_a.resolve(curves)?;
            Some(MediumEnum::HenyeyGreensteinHomogeneous(
                HenyeyGreensteinHomogeneous {
                    g,
                    sigma_s,
                    sigma_a,
                },
            ))
        }
        MediumData::Rayleigh(data) => {
            let ior = data.ior.resolve(curves)?;
            Some(MediumEnum::Rayleigh(Rayleigh::new(
                data.corrective_factor,
                ior,
            )))
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct NamedMedium {
    pub data: MediumData,
    pub name: String,
}
