use crate::math::*;
use crate::parsing::curves::CurveData;
use crate::renderer::Film;
use crate::texture::*;

use math::curves::InterpolationMode;
use math::spectral::BOUNDED_VISIBLE_RANGE;
use packed_simd::f32x4;
use serde::{Deserialize, Serialize};

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum TextureData {
    Texture1 {
        curve: CurveData,
        filename: String,
    },
    Texture4 {
        curves: [CurveData; 4],
        filename: String,
    },
    HDR {
        curves: [CurveData; 4],
        filename: String,
        alpha_fill: Option<f32>,
    },
    EXR {
        curves: [CurveData; 4],
        filename: String,
    },
    SRGB {
        filename: String,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TextureStackData {
    pub name: String,
    pub texture_stack: Vec<TextureData>,
}

pub fn parse_rgba(filepath: &str) -> Film<f32x4> {
    info!("parsing rgba texture at {}", filepath);
    let path = Path::new(filepath);
    let img = image::open(path).unwrap();
    let rgba_image = img.into_rgba8();
    let (width, height) = rgba_image.dimensions();
    let mut new_film = Film::new(width as usize, height as usize, f32x4::splat(0.0));
    for (x, y, pixel) in rgba_image.enumerate_pixels() {
        let [r, g, b, a]: [u8; 4] = pixel.0.into();
        new_film.write_at(
            x as usize,
            y as usize,
            f32x4::new(
                r as f32 / 256.0,
                g as f32 / 256.0,
                b as f32 / 256.0,
                a as f32 / 256.0,
            ),
        );
    }
    new_film
}

pub fn parse_exr(filepath: &str) -> Film<f32x4> {
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
            pixel_vector[position.y()][position.x()] = f32x4::new(r, g, b, a)
        },
    )
    .unwrap();

    let mut new_film = Film::new(
        rgba_image.layer_data.size.0 as usize,
        rgba_image.layer_data.size.1 as usize,
        f32x4::splat(0.0),
    );

    for (y, line) in rgba_image.layer_data.channel_data.pixels.iter().enumerate() {
        for (x, rgba) in line.iter().enumerate() {
            new_film.write_at(x, y, *rgba);
        }
    }
    new_film
}

pub fn parse_hdr(filepath: &str, alpha_fill: f32) -> Film<f32x4> {
    info!("parsing hdr texture at {}", filepath);
    let path = Path::new(filepath);
    let img = image::codecs::hdr::HdrDecoder::new(BufReader::new(
        File::open(path).expect("couldn't open hdr file"),
    ))
    .expect("couldn't construct hdr decoder for some reason");
    let (width, height) = (
        img.metadata().width as usize,
        img.metadata().height as usize,
    );
    let mut new_film = Film::new(width as usize, height as usize, f32x4::splat(0.0));
    for (idx, pixel) in img.read_image_hdr().unwrap().iter().enumerate() {
        let [r, g, b]: [f32; 3] = pixel.0;
        let (x, y) = (idx % width, idx / width);
        new_film.write_at(
            x as usize,
            y as usize,
            f32x4::new(r as f32, g as f32, b as f32, alpha_fill),
        );
    }
    new_film
}

pub fn parse_bitmap(filepath: &str) -> Film<f32> {
    info!("parsing greyscale texture at {}", filepath);
    let path = Path::new(filepath);
    let img = image::open(path).unwrap();
    let greyscale = img.into_luma8();
    let (width, height) = greyscale.dimensions();
    let mut new_film = Film::new(width as usize, height as usize, 0.0);
    for (x, y, pixel) in greyscale.enumerate_pixels() {
        let grey: [u8; 1] = pixel.0.into();
        new_film.write_at(x as usize, y as usize, grey[0] as f32 / 256.0);
    }
    new_film
}

pub fn select_on_film<T1, T2, F>(film: &Film<T1>, closure: F) -> Film<T2>
where
    F: FnMut(&T1) -> T2,
{
    Film {
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

pub fn parse_texture(texture: TextureData) -> Texture {
    match texture {
        TextureData::Texture1 { curve, filename } => {
            let cdf: CurveWithCDF = Curve::from(curve).to_cdf(BOUNDED_VISIBLE_RANGE, 100);
            Texture::Texture1(Texture1 {
                curve: cdf,
                texture: parse_bitmap(&filename),
                interpolation_mode: InterpolationMode::Nearest,
            })
        }
        TextureData::Texture4 { curves, filename } => {
            let cdfs: [CurveWithCDF; 4] = convert_to_array(
                curves
                    .iter()
                    .map(|curve| Curve::from(curve.clone()).to_cdf(BOUNDED_VISIBLE_RANGE, 100))
                    .collect(),
            );
            Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_rgba(&filename),
                interpolation_mode: InterpolationMode::Nearest,
            })
        }
        TextureData::HDR {
            curves,
            filename,
            alpha_fill,
        } => {
            let cdfs: [CurveWithCDF; 4] = convert_to_array(
                curves
                    .iter()
                    .map(|curve| Curve::from(curve.clone()).to_cdf(BOUNDED_VISIBLE_RANGE, 100))
                    .collect(),
            );
            Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_hdr(&filename, alpha_fill.unwrap_or(0.0)),
                interpolation_mode: InterpolationMode::Nearest,
            })
        }
        TextureData::EXR { curves, filename } => {
            let cdfs: [CurveWithCDF; 4] = convert_to_array(
                curves
                    .iter()
                    .map(|curve| Curve::from(curve.clone()).to_cdf(BOUNDED_VISIBLE_RANGE, 100))
                    .collect(),
            );
            Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_exr(&filename),
                interpolation_mode: InterpolationMode::Nearest,
            })
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
                    filename: curves_filename.clone(),
                    column: 3,
                    domain_mapping: None,
                    interpolation_mode: InterpolationMode::Cubic,
                })
                .to_cdf(BOUNDED_VISIBLE_RANGE, 100),
                Curve::from(CurveData::Flat { strength: 0.0 }).to_cdf(BOUNDED_VISIBLE_RANGE, 100),
            ];
            Texture::Texture4(Texture4 {
                curves: cdfs,
                texture: parse_rgba(&filename),
                interpolation_mode: InterpolationMode::Nearest,
            })
        }
    }
}

pub fn parse_texture_stack(tex_stack: TextureStackData) -> TexStack {
    let mut textures = Vec::new();
    for v in tex_stack.texture_stack.iter() {
        textures.push(parse_texture(v.clone()));
    }
    TexStack { textures }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_bitmap() {
        let film = parse_bitmap("data/textures/test.bmp");
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
        let film = parse_rgba("data/textures/test.png");
        let mut sum = f32x4::splat(0.0);
        let mut max = f32x4::splat(0.0);
        for v in film.buffer {
            sum += v;
            if v.sum() > max.sum() {
                max = v;
            }
        }
        println!(
            "avg value is {:?}",
            sum / film.width as f32 / film.height as f32
        );
        println!("max value is {:?}", max);
    }
    #[test]
    fn test_hdr() {
        let film = parse_hdr("data/textures/test.hdr", 0.0);
        let mut sum = f32x4::splat(0.0);
        let mut max = f32x4::splat(0.0);
        for v in film.buffer {
            sum += v;
            if v.sum() > max.sum() {
                max = v;
            }
        }
        println!(
            "avg value is {:?}",
            sum / film.width as f32 / film.height as f32
        );
        println!("max value is {:?}", max);
    }

    #[test]
    fn test_exr() {
        let film = parse_exr("data/textures/test.exr");
        let mut sum = f32x4::splat(0.0);
        let mut max = f32x4::splat(0.0);
        for v in film.buffer {
            sum += v;
            if v.sum() > max.sum() {
                max = v;
            }
        }
        println!(
            "avg value is {:?}",
            sum / film.width as f32 / film.height as f32
        );
        println!("max value is {:?}", max);
    }

    #[test]
    fn test_parse_texture() {
        let texture = parse_texture(TextureData::SRGB {
            filename: "data/textures/test.png".to_string(),
        });

        println!("{}", texture.eval_at(550.0, (0.5, 0.5)));
    }
    #[test]
    fn test_parse_texture_stack() {
        let texture = parse_texture_stack(TextureStackData {
            name: "stack".to_string(),
            texture_stack: vec![TextureData::Texture4 {
                curves: [
                    CurveData::SimpleSpike {
                        lambda: 400.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    },
                    CurveData::SimpleSpike {
                        lambda: 550.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    },
                    CurveData::SimpleSpike {
                        lambda: 600.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    },
                    CurveData::SimpleSpike {
                        lambda: 700.0,
                        left_taper: 100.0,
                        right_taper: 100.0,
                        strength: 0.25,
                    },
                ],
                filename: "data/textures/test.png".to_string(),
            }],
        });

        println!("{}", texture.eval_at(550.0, (0.5, 0.5)));
    }
}
