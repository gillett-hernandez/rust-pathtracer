use serde::Deserialize;

use crate::tonemap::{
    Clamp, Converter, Reinhard0, Reinhard0x3, Reinhard1, Reinhard1x3, Tonemapper,
};

#[derive(Deserialize, Clone, Copy)]
#[serde(tag = "type")]
pub enum TonemapSettings {
    // clamp all colors to 0 to 1, multiplying by 10^exposure beforehand (exposure defaults to 0, not changing anything)
    // high exposure implies low light levels, thus necessitating a boost
    // conversely, low exposure implies high light levels, requiring a downscaling
    Clamp {
        exposure: Option<f32>,
    },
    Reinhard0 {
        key_value: f32,
        luminance_only: bool,
    },
    Reinhard1 {
        key_value: f32,
        white_point: f32,
        luminance_only: bool,
    },
}

pub fn parse_tonemapper(settings: TonemapSettings) -> (Box<dyn Tonemapper>, Converter) {
    let tonemapper: Box<dyn Tonemapper> = match settings {
        TonemapSettings::Clamp { exposure } => Box::new(Clamp::new(exposure.unwrap_or(0.0))),
        TonemapSettings::Reinhard0 {
            key_value,
            luminance_only,
        } => {
            if luminance_only {
                Box::new(Reinhard0::new(key_value))
            } else {
                Box::new(Reinhard0x3::new(key_value))
            }
        }
        TonemapSettings::Reinhard1 {
            key_value,
            white_point,
            luminance_only,
        } => {
            if luminance_only {
                Box::new(Reinhard1::new(key_value, white_point))
            } else {
                Box::new(Reinhard1x3::new(key_value, white_point))
            }
        }
    };
    (tonemapper, Converter::sRGB)
}
