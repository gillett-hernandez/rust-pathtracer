use crate::mediums::*;
use crate::parsing::curves::{parse_curve, CurveData};
use crate::texture::TexStack;

use serde::{Deserialize, Serialize};

use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct HGMediumData {
    pub g: CurveData,
    pub sigma_s: CurveData,
    pub sigma_t: CurveData,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum MediumData {
    HG(HGMediumData),
}

pub fn parse_medium(data: MediumData) -> MediumEnum {
    match data {
        MediumData::HG(data) => {
            println!("parsing HG");
            let g = parse_curve(data.g);
            let sigma_s = parse_curve(data.sigma_s);
            let sigma_t = parse_curve(data.sigma_t);
            MediumEnum::HenyeyGreensteinHomogeneous(HenyeyGreensteinHomogeneous {
                g,
                sigma_s,
                sigma_t,
            })
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NamedMedium {
    pub data: MediumData,
    pub name: String,
}
