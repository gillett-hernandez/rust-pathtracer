use crate::prelude::*;

use math::curves::InterpolationMode;
use ordered_float::Float;

use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DomainMapping {
    pub x_offset: Option<f32>,
    pub x_scale: Option<f32>,
    pub y_offset: Option<f32>,
    pub y_scale: Option<f32>,
}

impl std::hash::Hash for DomainMapping {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x_offset.map(|e| e.integer_decode().hash(state));
        self.x_scale.map(|e| e.integer_decode().hash(state));
        self.y_offset.map(|e| e.integer_decode().hash(state));
        self.y_scale.map(|e| e.integer_decode().hash(state));
    }
}
impl Default for DomainMapping {
    // default domain mapping is the identity mapping. all inputs are unchanged upon being mapped.
    fn default() -> Self {
        DomainMapping {
            x_offset: Some(0.0),
            x_scale: Some(1.0),
            y_offset: Some(0.0),
            y_scale: Some(1.0),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum CurveData {
    Blackbody {
        temperature: f32,
        strength: f32,
    },
    Linear {
        filename: String,
        domain_mapping: Option<DomainMapping>,
        interpolation_mode: InterpolationMode,
    },
    TabulatedCSV {
        filename: String,
        column: usize,
        domain_mapping: Option<DomainMapping>,
        interpolation_mode: InterpolationMode,
    },
    Flat {
        strength: f32,
    },
    Cauchy {
        a: f32,
        b: f32,
    },
    SimpleSpike {
        lambda: f32,
        left_taper: f32,
        right_taper: f32,
        strength: f32,
    },
}

impl std::hash::Hash for CurveData {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            CurveData::Blackbody {
                temperature,
                strength,
            } => {
                temperature.integer_decode().hash(state);
                strength.integer_decode().hash(state);
            }
            CurveData::Linear {
                filename,
                domain_mapping,
                interpolation_mode,
            } => {
                filename.hash(state);
                domain_mapping.hash(state);
                core::mem::discriminant(interpolation_mode).hash(state);
            }
            CurveData::TabulatedCSV {
                filename,
                column,
                domain_mapping,
                interpolation_mode,
            } => {
                filename.hash(state);
                column.hash(state);
                domain_mapping.hash(state);
                core::mem::discriminant(interpolation_mode).hash(state);
            }
            CurveData::Flat { strength } => {
                strength.integer_decode().hash(state);
            }
            CurveData::Cauchy { a, b } => {
                a.integer_decode().hash(state);
                b.integer_decode().hash(state);
            }
            CurveData::SimpleSpike {
                lambda,
                left_taper,
                right_taper,
                strength,
            } => {
                lambda.integer_decode().hash(state);
                left_taper.integer_decode().hash(state);
                right_taper.integer_decode().hash(state);
                strength.integer_decode().hash(state);
            }
        }
    }
}

pub fn spectra(filename: &str, strength: f32) -> Curve {
    // defaults to cubic interpolation mode
    load_linear(filename, |x| x, |y| strength * y, InterpolationMode::Cubic)
        .unwrap_or_else(|_| panic!("failed parsing spectra file {}", filename))
}

pub fn parse_tabulated_curve_from_csv<F1, F2>(
    data: &str,
    column: usize,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    range_func: F2,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let mut signal: Vec<(f32, f32)> = Vec::new();
    for line in data.split_terminator('\n') {
        // if line.starts_with(pat)
        let mut split = line.split(',').take(column + 1);
        let x = split.next();
        for _ in 0..(column - 1) {
            let _ = split.next();
        }
        let y = split.next();

        if let (Some(a), Some(b)) = (x, y) {
            let (a2, b2) = (a.trim().parse::<f32>(), b.trim().parse::<f32>());
            if let (Ok(new_x), Ok(new_y)) = (a2, b2) {
                signal.push((domain_func(new_x), range_func(new_y)));
            } else {
                info!("skipped csv line {:?} {:?}", a, b);
                continue;
            }
        }
    }
    Ok(Curve::Tabulated {
        signal,
        mode: interpolation_mode,
    })
}

pub fn parse_linear<F1, F2>(
    data: &str,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    range_func: F2,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let mut lines = data.split_terminator('\n');
    let first_line = lines.next().unwrap();
    let mut split = first_line.split(',');
    let (start_x, step_size) = (split.next().unwrap(), split.next().unwrap());
    let (start_x, step_size) = (
        start_x.trim().parse::<f32>()?,
        step_size.trim().parse::<f32>()?,
    );
    info!(
        "parsing linear, (start_x, step_size) = ({}, {}) ",
        start_x, step_size
    );

    let mut values: Vec<f32> = Vec::new();
    for line in lines {
        let value = line.trim().parse::<f32>()?;
        values.push(range_func(value));
    }

    let end_x = start_x + step_size * (values.len() as f32);

    info!("end x = {}", end_x);

    Ok(Curve::Linear {
        signal: values,
        bounds: Bounds1D::new(domain_func(start_x), domain_func(end_x)),
        mode: interpolation_mode,
    })
}

pub fn load_ior_and_kappa<F>(filename: &str, func: F) -> Result<(Curve, Curve), Box<dyn Error>>
where
    F: Clone + Copy + Fn(f32) -> f32,
{
    let curves = load_multiple_csv_rows(filename, 2, InterpolationMode::Cubic, func, |y| y)?;
    Ok((curves[0].clone(), curves[1].clone()))
}

pub fn load_csv<F1, F2>(
    filename: &str,
    selected_column: usize,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    range_func: F2,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    assert!(selected_column > 0);
    let curve = parse_tabulated_curve_from_csv(
        buf.as_ref(),
        selected_column,
        interpolation_mode,
        domain_func,
        range_func,
    )?;
    Ok(curve)
}

pub fn load_multiple_csv_rows<F1, F2>(
    filename: &str,
    num_columns: usize,
    interpolation_mode: InterpolationMode,
    domain_func: F1,
    range_func: F2,
) -> Result<Vec<Curve>, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    let mut curves: Vec<Curve> = Vec::new();
    for column in 1..=num_columns {
        let curve = parse_tabulated_curve_from_csv(
            buf.as_ref(),
            column,
            interpolation_mode,
            domain_func,
            range_func,
        )?;
        curves.push(curve);
    }
    // let kappa = parse_tabulated_curve_from_csv(buf.as_ref(), 2, InterpolationMode::Cubic, func)?;
    Ok(curves)
}

pub fn load_linear<F1, F2>(
    filename: &str,
    domain_func: F1,
    range_func: F2,
    interpolation_mode: InterpolationMode,
) -> Result<Curve, Box<dyn Error>>
where
    F1: Clone + Copy + Fn(f32) -> f32,
    F2: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    let curve = parse_linear(&buf, interpolation_mode, domain_func, range_func)?;
    // let kappa = parse_tabulated_curve_from_csv(buf.as_ref(), 2, InterpolationMode::Cubic, func)?;
    Ok(curve)
}

impl From<CurveData> for Curve {
    fn from(curve: CurveData) -> Self {
        match curve {
            CurveData::Blackbody {
                temperature,
                strength,
            } => Curve::Blackbody {
                temperature,
                boost: strength,
            },
            CurveData::Linear {
                filename,
                domain_mapping,
                interpolation_mode,
            } => {
                let domain_mapping = domain_mapping.unwrap_or_default();
                info!("attempting to parse and load linear file at {:?}", filename);
                let maybe_curve = load_linear(
                    &filename,
                    |x| {
                        (x - domain_mapping.x_offset.unwrap_or(0.0))
                            * domain_mapping.x_scale.unwrap_or(1.0)
                    },
                    |y| {
                        (y - domain_mapping.y_offset.unwrap_or(0.0))
                            * domain_mapping.y_scale.unwrap_or(1.0)
                    },
                    interpolation_mode,
                );
                let curve = maybe_curve.expect("loading linear data failed");
                info!("parsed linear curve");
                curve
            }
            CurveData::Cauchy { a, b } => Curve::Cauchy { a, b },
            CurveData::TabulatedCSV {
                filename,
                column,
                domain_mapping,
                interpolation_mode,
            } => {
                let domain_mapping = domain_mapping.unwrap_or_default();
                info!("attempting to parse and load csv at file {:?}", filename);
                let maybe_curve = load_csv(
                    &filename,
                    column,
                    interpolation_mode,
                    |x| {
                        (x - domain_mapping.x_offset.unwrap_or(0.0))
                            * domain_mapping.x_scale.unwrap_or(1.0)
                    },
                    |y| {
                        (y - domain_mapping.y_offset.unwrap_or(0.0))
                            * domain_mapping.y_scale.unwrap_or(1.0)
                    },
                );
                let curve = maybe_curve.expect("loading tabulated data failed");
                info!("parsed tabulated curve");
                curve
            }
            CurveData::Flat { strength } => Curve::Linear {
                signal: vec![strength],
                bounds: EXTENDED_VISIBLE_RANGE,
                mode: InterpolationMode::Linear,
            },
            CurveData::SimpleSpike {
                lambda,
                left_taper,
                right_taper,
                strength,
            } => Curve::Exponential {
                signal: vec![(lambda, left_taper, right_taper, strength)],
            },
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum CurveDataOrReference {
    Literal(CurveData),
    Reference(String),
}
impl CurveDataOrReference {
    pub fn resolve(&self, curves_mapping: &HashMap<String, Curve>) -> Option<Curve> {
        match self {
            Self::Literal(inner) => {
                info!("resolving curve literal");
                Some(inner.clone().into())
            }
            Self::Reference(name) => {
                info!("resolving curve {}", name);
                curves_mapping.get(name).cloned()
            }
        }
    }
    pub fn get_name(&self) -> Option<&str> {
        match self {
            Self::Literal(_) => None,
            Self::Reference(name) => Some(name),
        }
    }
}

impl From<CurveData> for CurveDataOrReference {
    fn from(data: CurveData) -> Self {
        CurveDataOrReference::Literal(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_tabulated_curve() {
        let (gold_ior, gold_kappa) =
            load_ior_and_kappa("data/test/gold.csv", |x: f32| x * 1000.0).unwrap();

        println!("{:?}", gold_ior.evaluate_power(500.0));
        println!("{:?}", gold_kappa.evaluate_power(500.0));
    }
    #[test]
    fn test_parse_cornell() {
        let cornell_colors = load_multiple_csv_rows(
            "data/test/cornell.csv",
            3,
            InterpolationMode::Cubic,
            |x| x,
            |y| y,
        )
        .expect("data/test/cornell.csv was not formatted correctly");
        let mut iter = cornell_colors.iter();
        let (cornell_white, cornell_green, cornell_red) = (
            iter.next().unwrap().clone(),
            iter.next().unwrap().clone(),
            iter.next().unwrap().clone(),
        );

        println!(
            "{:?} {:?}",
            cornell_white.evaluate(520.0),
            cornell_white.evaluate(660.0)
        );
        println!(
            "{:?} {:?}",
            cornell_green.evaluate(520.0),
            cornell_green.evaluate(660.0)
        );
        println!(
            "{:?} {:?}",
            cornell_red.evaluate(520.0),
            cornell_red.evaluate(660.0)
        );

        println!(
            "{:?}",
            cornell_white.convert_to_xyz(Bounds1D::new(400.0, 700.0), 1.0, true)
        );

        println!(
            "{:?}",
            cornell_red.convert_to_xyz(Bounds1D::new(400.0, 700.0), 1.0, true)
        );

        println!(
            "{:?}",
            cornell_green.convert_to_xyz(Bounds1D::new(400.0, 700.0), 1.0, true)
        );
    }
    #[test]
    fn test_parse_linear_spectra() {
        let spectra = load_linear(
            "data/test/xenon_lamp.spectra",
            |x| x,
            |y| 10.0 * y,
            InterpolationMode::Cubic,
        )
        .unwrap();
        println!("{}", spectra.evaluate_power(500.0));
        println!("{}", spectra.evaluate_power(500.5));
    }
}
