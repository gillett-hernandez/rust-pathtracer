use crate::prelude::*;

use exr::{
    image::{write::WritableImage, AnyChannel, AnyChannels, Encoding, FlatSamples, Layer},
    math::Vec2,
    meta::{
        attribute::{Chromaticities, IntegerBounds},
        header::LayerAttributes,
    },
};
use nalgebra::{Matrix3, Vector3};

use std::{collections::HashMap, error::Error, marker::PhantomData, simd::f32x2, time::Instant};

mod clamp;
mod reinhard0;
mod reinhard1;

pub use clamp::Clamp;
pub use reinhard0::{Reinhard0, Reinhard0x3};
pub use reinhard1::{Reinhard1, Reinhard1x3};

// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
const XYZ_TO_REC709_LINEAR: Matrix3<f32> = Matrix3::new(
    3.24096994,
    -1.53738318,
    -0.49861076,
    -0.96924364,
    1.8759675,
    0.04155506,
    0.05563008,
    -0.20397696,
    1.05697151,
);

// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
const XYZ_TO_REC2020_LINEAR: Matrix3<f32> = Matrix3::new(
    1.4628067, -0.1840623, -0.2743606, -0.5217933, 1.4472381, 0.0677227, 0.0349342, -0.0968930,
    1.2884099,
);

pub trait Tonemapper: Send + Sync {
    // factor in `initialize` is a prefactor weight on the film contents.
    // a tonemapper should initialize based on the film x scale_factor
    // and should premultiply the film values when tonemapping through `map`
    fn initialize(&mut self, film: &Vec2D<XYZColor>, factor: f32);
    // should tonemap a pixel from hdr to ldr
    fn map(&self, film: &Vec2D<XYZColor>, pixel: (usize, usize)) -> XYZColor;
    fn get_name(&self) -> &str;
}

#[derive(Clone, Copy)]
pub struct Color<T: Default> {
    pub values: f32x4,
    color_space: PhantomData<*mut T>,
}

impl<T: Default> Color<T> {
    pub fn new(v: f32x4) -> Self {
        Self {
            values: v,
            color_space: PhantomData::default(),
        }
    }
}

pub struct Primaries {
    red: f32x2,
    green: f32x2,
    blue: f32x2,
    white: f32x2,
}

const REC709: Primaries = Primaries {
    // https://en.wikipedia.org/wiki/Rec._709#Primary_chromaticities
    // 0.3127 	0.3290 	0.64 	0.33 	0.30 	0.60 	0.15 	0.06
    red: f32x2::from_array([0.64, 0.33]),
    green: f32x2::from_array([0.30, 0.60]),
    blue: f32x2::from_array([0.15, 0.06]),
    white: f32x2::from_array([0.3127, 0.3290]),
};

const REC2020: Primaries = Primaries {
    // https://en.wikipedia.org/wiki/Rec._2020#System_colorimetry
    // 0.3127 	0.3290 	0.708 	0.292 	0.170 	0.797 	0.131 	0.046
    red: f32x2::from_array([0.708, 0.292]),
    green: f32x2::from_array([0.292, 0.170]),
    blue: f32x2::from_array([0.131, 0.046]),
    white: f32x2::from_array([0.3127, 0.3290]),
};

#[derive(Copy, Clone, Default)]
pub struct CIEXYZ;

#[derive(Copy, Clone, Default)]
pub struct CIERGB;

#[derive(Copy, Clone, Default)]
pub struct Rec709Primaries;

#[derive(Copy, Clone, Default)]
pub struct Rec2020Primaries;

impl From<XYZColor> for Color<CIEXYZ> {
    fn from(value: XYZColor) -> Self {
        Self {
            values: value.0,
            color_space: PhantomData::default(),
        }
    }
}

impl From<Color<CIEXYZ>> for Color<Rec709Primaries> {
    fn from(v: Color<CIEXYZ>) -> Color<Rec709Primaries> {
        let [x, y, z, _]: [f32; 4] = v.values.into();
        let intermediate = XYZ_TO_REC709_LINEAR * Vector3::new(x, y, z);

        Color::<Rec709Primaries>::new(f32x4::from_array([
            intermediate[0],
            intermediate[1],
            intermediate[2],
            0.0,
        ]))
    }
}

impl From<Color<CIEXYZ>> for Color<Rec2020Primaries> {
    fn from(v: Color<CIEXYZ>) -> Color<Rec2020Primaries> {
        let [x, y, z, _]: [f32; 4] = v.values.into();
        let intermediate = XYZ_TO_REC2020_LINEAR * Vector3::new(x, y, z);

        Color::<Rec2020Primaries>::new(f32x4::from_array([
            intermediate[0],
            intermediate[1],
            intermediate[2],
            0.0,
        ]))
    }
}

pub trait OETF {
    fn oetf(linear_color: f32x4) -> f32x4;
    fn primaries() -> Primaries;
    fn effective_gamma() -> f32;
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default)]
pub struct sRGB;

impl OETF for sRGB {
    fn oetf(linear_color: f32x4) -> f32x4 {
        (linear_color.simd_lt(f32x4::splat(0.0031308))).select(
            f32x4::splat(323.0 / 25.0) * linear_color,
            f32x4::splat(211.0 / 200.0) * linear_color.powf(f32x4::splat(5.0 / 12.0))
                - f32x4::splat(11.0 / 200.0),
        )
    }
    fn primaries() -> Primaries {
        REC709
    }
    fn effective_gamma() -> f32 {
        1.0 / 2.2
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default)]
pub struct Rec709;

impl OETF for Rec709 {
    fn oetf(linear_color: f32x4) -> f32x4 {
        linear_color.simd_lt(f32x4::splat(0.018053968510)).select(
            f32x4::splat(4.5) * linear_color,
            f32x4::splat(1.099296826) * linear_color.powf(f32x4::splat(0.45))
                - f32x4::splat(0.099296826),
        )
    }
    fn primaries() -> Primaries {
        REC709
    }
    fn effective_gamma() -> f32 {
        1.0 / 1.95
    }
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Default)]
pub struct Rec2020;

impl OETF for Rec2020 {
    fn oetf(linear_color: f32x4) -> f32x4 {
        linear_color.simd_lt(f32x4::splat(0.018053968510)).select(
            f32x4::splat(4.5) * linear_color,
            f32x4::splat(1.099296826) * linear_color.powf(f32x4::splat(0.45))
                - f32x4::splat(0.099296826),
        )
    }
    fn primaries() -> Primaries {
        REC2020
    }
    fn effective_gamma() -> f32 {
        1.0 / 2.4
    }
}

pub fn write_to_files<S, T>(
    film: &Vec2D<XYZColor>,
    tonemapper: &Box<dyn Tonemapper>,
    factor: f32,
    exr_filename: &str,
    png_filename: &str,
) -> Result<(), Box<dyn Error>>
where
    Color<S>: From<Color<CIEXYZ>>,
    S: Default,
    T: OETF + Default,
{
    let now = Instant::now();

    print!("saving exr...");

    let primaries = T::primaries();

    exr::prelude::write_rgb_file(exr_filename, film.width, film.height, |x, y| {
        let as_color: Color<CIEXYZ> = (factor * film.at(x, y)).into();
        let as_s: Color<S> = as_color.into();
        // Linear RGB color space with S primaries
        let [r, g, b, _]: [f32; 4] = as_s.values.into();
        (r, g, b)
    })?;

    let mut red = Vec::new();
    let mut green = Vec::new();
    let mut blue = Vec::new();

    for y in 0..film.height {
        for x in 0..film.width {
            let as_color: Color<CIEXYZ> = (factor * film.at(x, y)).into();
            let as_s: Color<S> = as_color.into();
            // Linear RGB color space with S primaries
            let [r, g, b, _]: [f32; 4] = as_s.values.into();

            red.push(r);
            green.push(g);
            blue.push(b);
        }
    }

    let image = exr::image::Image::empty(exr::prelude::ImageAttributes {
        display_window: IntegerBounds::new((0, 0), (film.width, film.height)),
        pixel_aspect: 1.0,
        chromaticities: Some(Chromaticities {
            red: Vec2(primaries.red[0], primaries.red[1]),
            green: Vec2(primaries.green[0], primaries.green[1]),
            blue: Vec2(primaries.blue[0], primaries.blue[1]),
            white: Vec2(primaries.white[0], primaries.white[1]),
        }),
        time_code: None,
        other: HashMap::new(),
    })
    .with_layer(Layer::new(
        // the only layer in this image
        (film.width, film.height),                // resolution
        LayerAttributes::named("main-rgb-layer"), // the layer has a name and other properties
        Encoding::FAST_LOSSLESS,                  // compress slightly
        AnyChannels::sort(smallvec![
            // the channels contain the actual pixel data
            AnyChannel::new("R", FlatSamples::F32(red)), // this channel contains all red values
            AnyChannel::new("G", FlatSamples::F32(green)), // this channel contains all green values
            AnyChannel::new("B", FlatSamples::F32(blue)), // this channel contains all blue values
        ]),
    ));
    let _ = image.write().to_file(exr_filename);
    println!(" done!");
    // let mut img: image::RgbImage = image::ImageBuffer::new(film.width as u32, film.height as u32);

    // for (x, y, pixel) in img.enumerate_pixels_mut() {
    //     //apply tonemap and T OETF here

    //     let [r, g, b, _]: [f32; 4] = T::oetf(tonemapper.map(film, (x as usize, y as usize))).into();

    //     *pixel = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
    // }
    // print!("saving image...");
    // img.save(png_filename)?;
    // println!(" done!");
    drop(image);

    use std::fs::File;
    use std::io::BufWriter;
    use std::path::Path;
    let path = Path::new(png_filename);
    let file = File::create(path).unwrap();
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(
        w,
        film.width.try_into().unwrap(),
        film.height.try_into().unwrap(),
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(T::effective_gamma())); // 1.0 / 2.2, unscaled, but rounded
    let source_chromaticities = png::SourceChromaticities::new(
        // Using unscaled instantiation here
        (primaries.white[0], primaries.white[1]),
        (primaries.red[0], primaries.red[1]),
        (primaries.green[0], primaries.green[1]),
        (primaries.blue[0], primaries.blue[1]),
    );
    encoder.set_source_chromaticities(source_chromaticities);
    let mut writer = encoder.write_header().unwrap();

    let mut data = Vec::new();
    for y in 0..film.height {
        for x in 0..film.width {
            let as_color: Color<CIEXYZ> = tonemapper.map(film, (x as usize, y as usize)).into();
            let as_s: Color<S> = as_color.into();
            // Linear RGB color space with S primaries
            let [r, g, b, _]: [f32; 4] = T::oetf(as_s.values.into()).into();

            data.extend_from_slice(&[
                (r * 255.0).ceil().clamp(0.0, 255.0) as u8,
                (g * 255.0).ceil().clamp(0.0, 255.0) as u8,
                (b * 255.0).ceil().clamp(0.0, 255.0) as u8,
                255,
            ]);
        }
    }

    writer.write_image_data(&data).unwrap();

    println!(
        "took {}s to tonemap and output\n",
        (now.elapsed().as_millis() as f32) / 1000.0
    );
    Ok(())
}

// should the tonemapper own a converter, or should a converter own the townmapper

#[cfg(test)]
mod test {

    use rand::random;

    use crate::tonemap::{Clamp, Reinhard0, Reinhard0x3, Reinhard1, Reinhard1x3, OETF};

    use super::*;

    fn printout<S, T>()
    where
        Color<S>: From<Color<CIEXYZ>>,
        S: Default,
        T: OETF + Default,
    {
        let mut film = Vec2D::new(32, 32, XYZColor::BLACK);

        let num_samples = 10;
        // estimated total energy per pixel is proportional to num_samples
        println!("adding {} samples per pixel", num_samples as usize);
        film.buffer.par_iter_mut().for_each(|px| {
            for _ in 0..(num_samples as usize) {
                *px += SingleWavelength::new_from_range(random::<f32>(), BOUNDED_VISIBLE_RANGE)
                    .replace_energy(1.0)
                    .into();
            }
        });

        for pixel in film.buffer.iter() {
            let as_color: Color<CIEXYZ> = (*pixel / num_samples as f32).into();
            let as_s: Color<S> = as_color.into();
            // Linear RGB color space with S primaries
            let [r, g, b, _]: [f32; 4] = T::oetf(as_s.values.into()).into();

            let sdr_rgb = (
                (r * 255.0).ceil().clamp(0.0, 255.0) as u8,
                (g * 255.0).ceil().clamp(0.0, 255.0) as u8,
                (b * 255.0).ceil().clamp(0.0, 255.0) as u8,
            );
            println!("{:?}", sdr_rgb);
        }
    }

    #[test]
    fn test_srgb_sdr() {
        printout::<Rec709Primaries, sRGB>();
    }
    #[test]
    fn test_rec709_sdr() {
        printout::<Rec709Primaries, Rec709>();
    }
    #[test]
    fn test_rec2020_sdr() {
        printout::<Rec2020Primaries, Rec2020>();
    }

    #[test]
    fn test_write_to_file() {
        let num_samples = random::<f32>() * 1000.0 + 1.0;

        let mut tonemappers: Vec<Box<dyn Tonemapper>> = vec![
            Box::new(Clamp::new(0.0, true, false)),
            Box::new(Clamp::new(0.0, false, false)),
            Box::new(Reinhard0::new(0.18, false)),
            Box::new(Reinhard0x3::new(0.18, false)),
            Box::new(Reinhard1::new(0.18, 10.0, false)),
            Box::new(Reinhard1x3::new(0.18, 10.0, false)),
        ];

        let mut film = Vec2D::new(1024, 1024, XYZColor::BLACK);

        // estimated total energy per pixel is proportional to num_samples
        println!("adding {} samples per pixel", num_samples as usize);
        film.buffer.par_iter_mut().for_each(|px| {
            for _ in 0..(num_samples as usize) {
                *px += SingleWavelength::new_from_range(random::<f32>(), BOUNDED_VISIBLE_RANGE)
                    .replace_energy(1.0)
                    .into();
            }
        });

        for (i, tonemapper) in tonemappers.iter_mut().enumerate() {
            let name = tonemapper.get_name();
            println!("tonemapper is {}", name);

            let exr_filename = format!("test_output_{}_{}_{}.exr", num_samples as usize, name, i);
            let png_filename = format!("test_output_{}_{}_{}.png", num_samples as usize, name, i);
            tonemapper.initialize(&film, 1.0 / num_samples);
            write_to_files::<Rec709Primaries, sRGB>(
                &film,
                tonemapper,
                1.0 / num_samples,
                &exr_filename,
                &png_filename,
            )
            .expect("failed to write files");
        }
    }
}
