use crate::mediums::*;
use crate::parsing::curves::CurveData;

use math::Curve;
use serde::{Deserialize, Serialize};

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
            let g = Curve::from(data.g);
            let sigma_s = data.sigma_s.into();
            let sigma_t = data.sigma_t.into();
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
