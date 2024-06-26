use crate::prelude::*;

use crate::parsing::curves::CurveData;

use crate::texture::*;

use anyhow::{bail, Context};
use image::DynamicImage;
use math::curves::InterpolationMode;

use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use super::CurveDataOrReference;

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum TextureData {
    Texture1 {
        curve: CurveDataOrReference,
        filename: String,
    },
    Texture4 {
        curves: [CurveDataOrReference; 4],
        filename: String,
    },
    HDR {
        curves: [CurveDataOrReference; 4],
        filename: String,
        alpha_fill: Option<f32>,
    },
    EXR {
        curves: [CurveDataOrReference; 4],
        filename: String,
    },
    SRGB {
        filename: String,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TextureStackData(pub Vec<TextureData>);

pub fn parse_rgba(filepath: &str) -> anyhow::Result<Vec2D<f32x4>> {
    info!("parsing rgba texture at {}", filepath);
    let path = Path::new(filepath);

    let img = image::open(path).context(format!("could not find file at {}", filepath))?;
    let rgba_image = img.into_rgba8();
    let (width, height) = rgba_image.dimensions();
    let mut new_film = Vec2D::new(width as usize, height as usize, f32x4::splat(0.0));
    for (x, y, pixel) in rgba_image.enumerate_pixels() {
        // apparently, no into is needed since an rgba<u8> is just an alias for [u8;4]

        let [r, g, b, a]: [u8; 4] = pixel.0;
        new_film.write_at(
            x as usize,
            y as usize,
            f32x4::from_array([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ]),
        );
    }
    Ok(new_film)
}

pub fn parse_exr(filepath: &str) -> anyhow::Result<Vec2D<f32x4>> {
    info!("parsing exr texture at {}", filepath);
    let rgba_image = exr::prelude::read_first_rgba_layer_from_file(
        filepath,
        |resolution, _| {
            let default_pixel = f32x4::splat(0.0);
            let empty_line = vec![default_pixel; resolution.width()];
            let empty_image = vec![empty_line; resolution.height()];
            empty_image
        },
        |pixel_vector, position, (r, g, b, a): (f32, f32, f32, f32)| {
            pixel_vector[position.y()][position.x()] = f32x4::from_array([r, g, b, a])
        },
    )
    .context("failed to read exr layer from file")?;

    let mut new_film = Vec2D::new(
        rgba_image.layer_data.size.0 as usize,
        rgba_image.layer_data.size.1 as usize,
        f32x4::splat(0.0),
    );

    for (y, line) in rgba_image.layer_data.channel_data.pixels.iter().enumerate() {
        for (x, rgba) in line.iter().enumerate() {
            new_film.write_at(x, y, *rgba);
        }
    }
    Ok(new_film)
}

pub fn parse_hdr(filepath: &str, alpha_fill: f32) -> anyhow::Result<Vec2D<f32x4>> {
    info!("parsing hdr texture at {}", filepath);
    let path = Path::new(filepath);
    let hdr_decoder = image::codecs::hdr::HdrDecoder::new(BufReader::new(
        File::open(path).expect(&*format!("couldn't open hdr file at {}", filepath)),
    ))
    .context("couldn't construct hdr decoder for some reason")?;
    let (width, height) = (
        hdr_decoder.metadata().width as usize,
        hdr_decoder.metadata().height as usize,
    );
    let mut new_film = Vec2D::new(width, height, f32x4::splat(0.0));

    let hdr = match DynamicImage::from_decoder(hdr_decoder).unwrap() {
        DynamicImage::ImageRgb32F(image) => image,
        _ => bail!("expected rgb32f image"),
    };

    for (x, y, pixel) in hdr.enumerate_pixels() {
        let [r, g, b]: [f32; 3] = pixel.0;

        new_film.write_at(
            x as usize,
            y as usize,
            f32x4::from_array([r, g, b , alpha_fill]),
        );
    }
    Ok(new_film)
}

pub fn parse_bitmap(filepath: &str) -> anyhow::Result<Vec2D<f32>> {
    info!("parsing greyscale texture at {}", filepath);
    let path = Path::new(filepath);
    let img = image::open(path).context(format!("failed to open image at path {:?}", path))?;
    let greyscale = img.into_luma8();
    let (width, height) = greyscale.dimensions();
    let mut new_film = Vec2D::new(width as usize, height as usize, 0.0);
    for (x, y, pixel) in greyscale.enumerate_pixels() {
        let grey: [u8; 1] = pixel.0;
        new_film.write_at(x as usize, y as usize, grey[0] as f32 / 255.0);
    }
    Ok(new_film)
}

pub fn select_on_film<T1, T2, F>(film: &Vec2D<T1>, closure: F) -> Vec2D<T2>
where
    F: FnMut(&T1) -> T2,
{
    Vec2D {
        buffer: film.buffer.iter().map(closure).collect(),
        width: film.width,
        height: film.height,
    }
}

fn convert_to_array(vec: Vec<CurveWithCDF>) -> [CurveWithCDF; 4] {
    let mut arr: [CurveWithCDF; 4] = [
        CurveWithCDF::default(),
        CurveWithCDF::default(),
        CurveWithCDF::default(),
        CurveWithCDF::default(),
    ];
    arr[0] = vec[0].clone();
    arr[1] = vec[1].clone();
    arr[2] = vec[2].clone();
    arr[3] = vec[3].clone();
    arr
}

fn parse_curves(
    curves: &[CurveDataOrReference; 4],
    curves_data: &HashMap<String, Curve>,
) -> anyhow::Result<Vec<CurveWithCDF>> {
    let maybe_curves: Vec<Option<CurveWithCDF>> = curves
        .iter()
        .map(|curve| {
            curve
                .resolve(curves_data)
                .map(|some| some.to_cdf(BOUNDED_VISIBLE_RANGE, 100))
        })
        .collect();
    if let Some((index, _)) = maybe_curves.iter().enumerate().find(|e| e.1.is_none()) {
        let message = format!(
            "failed to parse the requisite number of curves, curve name {}",
            match &curves[index] {
                CurveDataOrReference::Reference(name) => {
                    name
                }
                _ => {
                    unreachable!()
                }
            }
        );
        error!("{}", message);
        bail!(message)
    } else {
        Ok(maybe_curves.into_iter().map(|e| e.unwrap()).collect::<_>())
    }
}

pub fn parse_texture(
    texture: TextureData,
    curves_data: &HashMap<String, Curve>,
) -> anyhow::Result<Texture> {
    match texture {
        TextureData::Texture1 { curve, filename } => {
            let cdf: CurveWithCDF = curve
                .resolve(curves_data)
                .context("failed to resolve curve reference")?
                .to_cdf(BOUNDED_VISIBLE_RANGE, 100);
            Ok(Texture::Texture1(Texture1 {
                curve: cdf,
                texture: parse_bitmap(&filename)?,
                interpolation_mode: InterpolationMode::Nearest,
            }))
        }
        TextureData::Texture4 { curves, filename } => {
            let curves = parse_curves(&curves, curves_data).context("failed to parse curve")?;
            let cdfs: [CurveWithCDF; 4] = convert_to_array(curves);
            Ok(Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_rgba(&filename)?,
                interpolation_mode: InterpolationMode::Nearest,
            }))
        }
        TextureData::HDR {
            curves,
            filename,
            alpha_fill,
        } => {
            let curves = parse_curves(&curves, curves_data).context("failed to parse curve")?;
            let cdfs: [CurveWithCDF; 4] = convert_to_array(curves);

            Ok(Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_hdr(&filename, alpha_fill.unwrap_or(0.0))?,
                interpolation_mode: InterpolationMode::Nearest,
            }))
        }
        TextureData::EXR { curves, filename } => {
            let curves = parse_curves(&curves, curves_data).context("failed to parse curve")?;
            let cdfs: [CurveWithCDF; 4] = convert_to_array(curves);
            Ok(Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_exr(&filename)?,
                interpolation_mode: InterpolationMode::Nearest,
            }))
        }
        TextureData::SRGB { filename } => {
            let curves_filename = "data/curves/basis/simple-spectral-srgb-1931.csv".to_string();
            let cdfs: [CurveWithCDF; 4] = [
                Curve::from(CurveData::TabulatedCSV {
                    filename: curves_filename.clone(),
                    column: 1,
                    domain_mapping: None,
                    interpolation_mode: InterpolationMode::Cubic,
                })
                .to_cdf(BOUNDED_VISIBLE_RANGE, 100),
                Curve::from(CurveData::TabulatedCSV {
                    filename: curves_filename.clone(),
                    column: 2,
                    domain_mapping: None,
                    interpolation_mode: InterpolationMode::Cubic,
                })
                .to_cdf(BOUNDED_VISIBLE_RANGE, 100),
                Curve::from(CurveData::TabulatedCSV {
                    filename: curves_filename,
                    column: 3,
                    domain_mapping: None,
                    interpolation_mode: InterpolationMode::Cubic,
                })
                .to_cdf(BOUNDED_VISIBLE_RANGE, 100),
                Curve::from(CurveData::Flat { strength: 0.0 }).to_cdf(BOUNDED_VISIBLE_RANGE, 100),
            ];
            Ok(Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_rgba(&filename).context("failed to parse rgba")?,
                interpolation_mode: InterpolationMode::Nearest,
            }))
        }
    }
}

pub fn parse_texture_stack(
    tex_stack: TextureStackData,
    curves_data: &HashMap<String, Curve>,
) -> anyhow::Result<TexStack> {
    let mut textures = Vec::new();
    for v in tex_stack.0.iter() {
        textures.push(parse_texture(v.clone(), curves_data)?);
    }
    Ok(TexStack { textures })
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_bitmap() {
        let film = parse_bitmap("data/test/test.bmp").expect("failed to parse test bmp");
        let mut sum = 0.0;
        let mut max = 0.0;
        for v in film.buffer {
            sum += v;
            if v > max {
                max = v;
            }
        }
        println!(
            "avg value is {}",
            sum / film.width as f32 / film.height as f32
        );
        println!("max value is {}", max);
    }
    #[test]
    fn test_rgba() {
        let film = parse_rgba("data/test/test.png").expect("failed to parse test png");
        let mut sum = f32x4::splat(0.0);
        let mut max = f32x4::splat(0.0);
        for v in film.buffer {
            sum += v;
            if v.reduce_sum() > max.reduce_sum() {
                max = v;
            }
        }
        println!(
            "avg value is {:?}",
            sum / f32x4::splat(film.width as f32 * film.height as f32)
        );
        println!("max value is {:?}", max);
    }
    #[test]
    fn test_hdr() {
        let film = parse_hdr("data/test/test.hdr", 0.0).expect("failed to parse test hdr");
        let mut sum = f32x4::splat(0.0);
        let mut max = f32x4::splat(0.0);
        for v in film.buffer {
            sum += v;
            if v.reduce_sum() > max.reduce_sum() {
                max = v;
            }
        }
        println!(
            "avg value is {:?}",
            sum / f32x4::splat(film.width as f32 * film.height as f32)
        );
        println!("max value is {:?}", max);
    }

    #[test]
    fn test_exr() {
        let film = parse_exr("data/test/test.exr").expect("failed to parse test exr");
        let mut sum = f32x4::splat(0.0);
        let mut max = f32x4::splat(0.0);
        for v in film.buffer {
            sum += v;
            if v.reduce_sum() > max.reduce_sum() {
                max = v;
            }
        }
        println!(
            "avg value is {:?}",
            sum / f32x4::splat(film.width as f32 * film.height as f32)
        );
        println!("max value is {:?}", max);
    }

    #[test]
    fn test_parse_texture() {
        let map = HashMap::new();
        let texture = parse_texture(
            TextureData::SRGB {
                filename: "data/test/test.png".to_string(),
            },
            &map,
        )
        .unwrap();

        println!("{}", texture.eval_at(550.0, (0.5, 0.5).into()));
    }
    #[test]
    fn test_parse_texture_stack() {
        let map = HashMap::new();
        let texture = parse_texture_stack(
            TextureStackData(vec![TextureData::Texture4 {
                curves: [
                    CurveData::SimpleSpike {
                        lambda: 400.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    }
                    .into(),
                    CurveData::SimpleSpike {
                        lambda: 550.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    }
                    .into(),
                    CurveData::SimpleSpike {
                        lambda: 600.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    }
                    .into(),
                    CurveData::SimpleSpike {
                        lambda: 700.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    }
                    .into(),
                ],
                filename: "data/test/test.png".to_string(),
            }]),
            &map,
        )
        .unwrap();

        println!("{}", texture.eval_at(550.0, (0.5, 0.5).into()));
    }
}
