use crate::math::*;
use crate::parsing::curves::{parse_curve, CurveData};
use crate::renderer::Film;
use crate::texture::*;

use packed_simd::f32x4;
use serde::{Deserialize, Serialize};

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
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TextureStackData {
    pub name: String,
    pub texture_stack: Vec<TextureData>,
}

pub fn parse_rgba(filepath: &str) -> Film<f32x4> {
    println!("parsing rgba texture at {}", filepath);
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

pub fn parse_bitmap(filepath: &str) -> Film<f32> {
    println!("parsing greyscale texture at {}", filepath);
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

pub fn select_channel(film: &Film<f32x4>, channel: usize) -> Film<f32> {
    assert!(channel < 4);

    Film {
        buffer: film.buffer.iter().map(|v| v.extract(channel)).collect(),
        width: film.width,
        height: film.height,
    }
}

fn convert_to_array(vec: Vec<CDF>) -> [CDF; 4] {
    let mut arr: [CDF; 4] = [
        CDF::default(),
        CDF::default(),
        CDF::default(),
        CDF::default(),
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
            let cdf: CDF = parse_curve(curve).into();
            Texture::Texture1(Texture1 {
                curve: cdf,
                texture: parse_bitmap(&filename),
                interpolation_mode: InterpolationMode::Nearest,
            })
        }
        TextureData::Texture4 { curves, filename } => {
            let cdfs: [CDF; 4] = convert_to_array(
                curves
                    .iter()
                    .map(|curve| parse_curve(curve.clone()).into())
                    .collect(),
            );
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
