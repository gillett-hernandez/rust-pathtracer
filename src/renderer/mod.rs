use crate::prelude::*;

mod film;
mod prelude;

// integrators
mod naive;
mod preview;
// mod sppm;

pub use film::Film;

pub use naive::NaiveRenderer;
pub use preview::PreviewRenderer;
// pub use sppm::SPPMRenderer;

use crate::camera::CameraEnum;

use crate::parsing::config::{Config, RenderSettings};
use crate::parsing::parse_tonemapper;
use crate::world::World;

pub fn output_film(render_settings: &RenderSettings, film: &Film<XYZColor>, factor: f32) {
    assert!(factor > 0.0);
    let filename = render_settings.filename.as_ref();
    let filename_str = filename.cloned().unwrap_or_else(|| String::from("output"));
    let exr_filename = format!("output/{}.exr", filename_str);
    let png_filename = format!("output/{}.png", filename_str);

    let (mut tonemapper, converter) = parse_tonemapper(render_settings.tonemap_settings);
    tonemapper.initialize(film, factor);

    if let Err(inner) = converter.write_to_files(film, &tonemapper, &exr_filename, &png_filename) {
        error!("failed to write files");
        error!("{:?}", inner.to_string());
        panic!();
    }
}

pub fn calculate_widest_wavelength_bounds(
    config: &[RenderSettings],
    default: Bounds1D,
) -> Bounds1D {
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
    info!("parsed wavelength bounds to be {:?}", result);
    result
}

pub trait Renderer {
    fn render(&self, world: World, cameras: Vec<CameraEnum>, config: &Config);
}

#[cfg(test)]
mod test {
    use rand::{random, seq::SliceRandom};

    use crate::{
        parsing::config::Resolution,
        tonemap::{Clamp, Reinhard0, Reinhard0x3, Reinhard1, Reinhard1x3},
    };

    use super::*;
    #[test]
    fn test_write_to_file() {
        let num_samples = random::<f32>() * 100.0 + 1.0;

        let mut tonemappers: Vec<Box<dyn Tonemapper>> = vec![
            Box::new(Clamp::new(0.0, true, false)),
            Box::new(Clamp::new(0.0, false, false)),
            Box::new(Reinhard0::new(0.18, false)),
            Box::new(Reinhard0x3::new(0.18, false)),
            Box::new(Reinhard1::new(0.18, 10.0, false)),
            Box::new(Reinhard1x3::new(0.18, 10.0, false)),
        ];

        let mut film = Film::new(1024, 1024, XYZColor::BLACK);

        // estimated total energy per pixel is proportional to num_samples
        println!("adding {} samples per pixel", num_samples as usize);
        film.buffer.par_iter_mut().for_each(|px| {
            for _ in 0..(num_samples as usize) {
                *px += SingleWavelength::new_from_range(random::<f32>(), BOUNDED_VISIBLE_RANGE)
                    .replace_energy(1.0)
                    .into();
            }
        });

        let converter = Converter::sRGB;
        for (i, tonemapper) in tonemappers.iter_mut().enumerate() {
            let name = tonemapper.get_name();
            println!("tonemapper is {}", name);

            let exr_filename = format!("test_output_{}_{}.exr", name, i);
            let png_filename = format!("test_output_{}_{}.png", name, i);
            tonemapper.initialize(&film, 1.0 / num_samples);
            converter
                .write_to_files(&film, tonemapper, &exr_filename, &png_filename)
                .expect("failed to write files");
        }
    }
}
