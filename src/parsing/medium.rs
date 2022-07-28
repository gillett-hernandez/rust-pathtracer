use std::collections::HashMap;

use crate::mediums::*;

use math::Curve;
use serde::{Deserialize, Serialize};

use super::CurveDataOrReference;

#[derive(Serialize, Deserialize, Clone)]
pub struct HGMediumData {
    pub g: CurveDataOrReference,
    pub sigma_s: CurveDataOrReference,
    pub sigma_t: CurveDataOrReference,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum MediumData {
    HG(HGMediumData),
}

pub fn parse_medium(data: MediumData, curves: &HashMap<String, Curve>) -> Option<MediumEnum> {
    match data {
        MediumData::HG(data) => {
            info!("parsing HG");
            let g = data.g.resolve(curves)?;
            let sigma_s = data.sigma_s.resolve(curves)?;
            let sigma_t = data.sigma_t.resolve(curves)?;
            Some(MediumEnum::HenyeyGreensteinHomogeneous(
                HenyeyGreensteinHomogeneous {
                    g,
                    sigma_s,
                    sigma_t,
                },
            ))
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NamedMedium {
    pub data: MediumData,
    pub name: String,
}
