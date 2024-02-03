use serde::Deserialize;

use crate::tonemap::{Clamp, Color, Reinhard0, Reinhard0x3, Reinhard1, Reinhard1x3, Tonemapper};

#[derive(Deserialize, Clone, Copy)]
#[serde(tag = "type")]
pub enum TonemapSettings {
    // clamp all colors to 0 to 1, multiplying by 10^exposure beforehand (exposure defaults to 0, not changing anything)
    // high exposure implies low light levels, thus necessitating a boost
    // conversely, low exposure implies high light levels, requiring a downscaling
    Clamp {
        exposure: Option<f32>,
        luminance_only: bool,
        #[serde(default)]
        silenced: bool,
    },
    Reinhard0 {
        key_value: f32,
        luminance_only: bool,
        #[serde(default)]
        silenced: bool,
    },
    Reinhard1 {
        key_value: f32,
        white_point: f32,
        luminance_only: bool,
        #[serde(default)]
        silenced: bool,
    },
}

impl TonemapSettings {
    pub fn silenced(self) -> Self {
        match self {
            TonemapSettings::Clamp {
                exposure,
                luminance_only,
                ..
            } => TonemapSettings::Clamp {
                exposure,
                luminance_only,
                silenced: true,
            },
            TonemapSettings::Reinhard0 {
                key_value,
                luminance_only,
                ..
            } => TonemapSettings::Reinhard0 {
                key_value,
                luminance_only,
                silenced: true,
            },
            TonemapSettings::Reinhard1 {
                key_value,
                white_point,
                luminance_only,
                ..
            } => TonemapSettings::Reinhard1 {
                key_value,
                white_point,
                luminance_only,
                silenced: true,
            },
        }
    }
}

pub fn parse_tonemap_settings(settings: TonemapSettings) -> Box<dyn Tonemapper> {
    let tonemapper: Box<dyn Tonemapper> = match settings {
        TonemapSettings::Clamp {
            exposure,
            luminance_only,
            silenced,
        } => Box::new(Clamp::new(
            exposure.unwrap_or(0.0),
            luminance_only,
            silenced,
        )),
        TonemapSettings::Reinhard0 {
            key_value,
            luminance_only,
            silenced,
        } => {
            if luminance_only {
                Box::new(Reinhard0::new(key_value, silenced))
            } else {
                Box::new(Reinhard0x3::new(key_value, silenced))
            }
        }
        TonemapSettings::Reinhard1 {
            key_value,
            white_point,
            luminance_only,
            silenced,
        } => {
            if luminance_only {
                Box::new(Reinhard1::new(key_value, white_point, silenced))
            } else {
                Box::new(Reinhard1x3::new(key_value, white_point, silenced))
            }
        }
    };
    tonemapper
}
