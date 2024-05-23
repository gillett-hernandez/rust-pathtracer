use crate::prelude::*;
use crate::tonemap::{sRGB, write_to_files, Rec2020, Rec2020Primaries, Rec709, Rec709Primaries};

mod prelude;

// integrators
mod naive;
mod preview;
// mod sppm;
mod tiled;

pub use naive::NaiveRenderer;
#[cfg(feature = "preview")]
pub use preview::PreviewRenderer;
pub use tiled::TiledRenderer;


use crate::parsing::config::{ColorSpaceSettings, Config, RenderSettings};
use crate::parsing::parse_tonemap_settings;
use crate::world::World;

use self::prelude::IntegratorKind;

pub fn output_film(render_settings: &RenderSettings, film: &Vec2D<XYZColor>, mut factor: f32) {
    factor *= render_settings.premultiply.unwrap_or(1.0);
    assert!(factor > 0.0);
    let filename = render_settings.filename.as_ref();
    let filename_str = filename.cloned().unwrap_or_else(|| String::from("beauty"));

    let exr_filename = format!("output/{}.exr", filename_str);
    let png_filename = format!("output/{}.png", filename_str);

    let mut tonemapper = parse_tonemap_settings(render_settings.tonemap_settings);
    tonemapper.initialize(film, factor);

    match render_settings.colorspace_settings {
        ColorSpaceSettings::sRGB => {
            info!("writing png as sRGB");
            if let Err(inner) = write_to_files::<Rec709Primaries, sRGB>(
                film,
                &tonemapper,
                factor,
                &exr_filename,
                &png_filename,
            ) {
                error!("failed to write files");
                error!("{:?}", inner.to_string());
                panic!();
            }
        }
        ColorSpaceSettings::Rec709 => {
            info!("writing png as Rec709");
            if let Err(inner) = write_to_files::<Rec709Primaries, Rec709>(
                film,
                &tonemapper,
                factor,
                &exr_filename,
                &png_filename,
            ) {
                error!("failed to write files");
                error!("{:?}", inner.to_string());
                panic!();
            }
        }
        ColorSpaceSettings::Rec2020 => {
            info!("writing png as Rec2020");
            if let Err(inner) = write_to_files::<Rec2020Primaries, Rec2020>(
                film,
                &tonemapper,
                factor,
                &exr_filename,
                &png_filename,
            ) {
                error!("failed to write files");
                error!("{:?}", inner.to_string());
                panic!();
            }
        }
    }
}

pub fn calculate_widest_wavelength_bounds(
    config: &[RenderSettings],
    default: Bounds1D,
) -> Bounds1D {
    let mut wavelength_bounds = None;
    for settings in config.iter() {
        if let Some((lower, upper)) = settings.wavelength_bounds {
            if wavelength_bounds.is_none() {
                wavelength_bounds = Some(Bounds1D::new(lower, upper));
            } else {
                let mut modified_bounds = wavelength_bounds.take().unwrap();
                modified_bounds.lower = modified_bounds.lower.min(lower);
                modified_bounds.upper = modified_bounds.upper.max(lower);
                wavelength_bounds = Some(modified_bounds);
            }
        }
    }
    let result = match wavelength_bounds {
        Some(bounds) => bounds,
        None => default,
    };
    info!("parsed wavelength bounds to be {:?}", result);
    result
}

pub trait Renderer {
    fn render(&self, world: World, config: &Config);
    fn supported_integrators(&self) -> &[IntegratorKind] {
        &[]
    }
}
