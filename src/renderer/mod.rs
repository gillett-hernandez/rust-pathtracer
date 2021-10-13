extern crate pbr;

mod film;
mod gpu_style;
mod naive;
mod preview;
mod sppm;

pub use film::Film;
pub use gpu_style::GPUStyleRenderer;
pub use naive::NaiveRenderer;
pub use preview::PreviewRenderer;
pub use sppm::SPPMRenderer;

use crate::camera::Camera;

use crate::config::{Config, RenderSettings};
use crate::math::{Bounds1D, XYZColor};
use crate::tonemap::{sRGB, Tonemapper};
use crate::world::World;

pub fn output_film(render_settings: &RenderSettings, film: &Film<XYZColor>) {
    let filename = render_settings.filename.as_ref();
    let filename_str = filename.cloned().unwrap_or(String::from("output"));
    let exr_filename = format!("output/{}.exr", filename_str);
    let png_filename = format!("output/{}.png", filename_str);

    let srgb_tonemapper = sRGB::new(film, render_settings.exposure.unwrap_or(1.0));
    srgb_tonemapper.write_to_files(film, &exr_filename, &png_filename);
}

pub fn parse_wavelength_bounds(config: &Vec<RenderSettings>, default: Bounds1D) -> Bounds1D {
    let mut wavelength_bounds: Option<Bounds1D> = None;
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
    println!("parsed wavelength bounds to be {:?}", result);
    result
}

pub trait Renderer {
    fn render(&self, world: World, cameras: Vec<Camera>, config: &Config);
}
