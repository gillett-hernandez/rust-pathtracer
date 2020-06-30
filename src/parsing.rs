use crate::math::*;
pub use crate::spectral::InterpolationMode;

use std::error::Error;
use std::fs::File;
use std::io::Read;
// use std::env;
// use std::io::{self, BufWriter, Write};
use std::path::Path;

pub fn parse_tabulated_curve_from_csv<F>(
    data: &str,
    column: usize,
    interpolation_mode: InterpolationMode,
    func: F,
) -> Result<SPD, Box<dyn Error>>
where
    F: Fn(f32) -> f32,
{
    let mut signal: Vec<(f32, f32)> = Vec::new();
    for line in data.split_terminator("\n") {
        // if line.starts_with(pat)
        let mut split = line.split(",").take(column + 1);
        let x = split.next();
        for _ in 0..(column - 1) {
            let _ = split.next();
        }
        let y = split.next();
        match (x, y) {
            (Some(a), Some(b)) => {
                let (a2, b2) = (a.trim().parse::<f32>(), b.trim().parse::<f32>());
                if let (Ok(new_x), Ok(new_y)) = (a2, b2) {
                    signal.push((func(new_x), new_y));
                } else {
                    println!("skipped csv line {:?} {:?}", a, b);
                    continue;
                }
            }
            _ => {}
        }
    }
    Ok(SPD::Tabulated {
        signal,
        mode: interpolation_mode,
    })
}

pub fn parse_linear<F>(
    data: &str,
    interpolation_mode: InterpolationMode,
    func: F,
) -> Result<SPD, Box<dyn Error>>
where
    F: Fn(f32) -> f32,
{
    let mut lines = data.split_terminator("\n");
    let first_line = lines.next().unwrap();
    let mut split = first_line.split(",");
    let (start_x, step_size) = (split.next().unwrap(), split.next().unwrap());
    let (start_x, step_size) = (
        start_x.trim().parse::<f32>()?,
        step_size.trim().parse::<f32>()?,
    );
    println!("{} {} ", start_x, step_size);

    let mut values: Vec<f32> = Vec::new();
    for line in lines {
        let value = line.trim().parse::<f32>()?;
        // print!("{} ", value);
        values.push(func(value));
    }

    let end_x = start_x + step_size * (values.len() as f32);

    println!("{}", end_x);

    Ok(SPD::Linear {
        signal: values,
        bounds: Bounds1D::new(start_x, end_x),
        mode: interpolation_mode,
    })
}

pub fn load_ior_and_kappa<F>(filename: &str, func: F) -> Result<(SPD, SPD), Box<dyn Error>>
where
    F: Clone + Copy + Fn(f32) -> f32,
{
    let curves = load_csv(filename, 2, func)?;
    Ok((curves[0].clone(), curves[1].clone()))
}

pub fn load_csv<F>(filename: &str, num_columns: usize, func: F) -> Result<Vec<SPD>, Box<dyn Error>>
where
    F: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    let mut curves: Vec<SPD> = Vec::new();
    for column in 1..=num_columns {
        let curve =
            parse_tabulated_curve_from_csv(buf.as_ref(), column, InterpolationMode::Cubic, func)?;
        curves.push(curve);
    }
    // let kappa = parse_tabulated_curve_from_csv(buf.as_ref(), 2, InterpolationMode::Cubic, func)?;
    Ok(curves)
}

pub fn load_linear<F>(filename: &str, func: F) -> Result<SPD, Box<dyn Error>>
where
    F: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    let curve = parse_linear(&buf, InterpolationMode::Cubic, func)?;
    // let kappa = parse_tabulated_curve_from_csv(buf.as_ref(), 2, InterpolationMode::Cubic, func)?;
    Ok(curve)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_tabulated_curve() {
        let (gold_ior, gold_kappa) =
            load_ior_and_kappa("data/curves/csv/gold.csv", |x: f32| x * 1000.0).unwrap();

        println!("{:?}", gold_ior.evaluate_power(500.0));
        println!("{:?}", gold_kappa.evaluate_power(500.0));
    }
    #[test]
    fn test_parse_cornell() {
        let cornell_colors = load_csv("data/curves/csv/cornell.csv", 3, |x| x)
            .expect("data/curves/csv/cornell.csv was not formatted correctly");
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
            cornell_white.convert_to_xyz(Bounds1D::new(400.0, 700.0), 1.0)
        );

        println!(
            "{:?}",
            cornell_red.convert_to_xyz(Bounds1D::new(400.0, 700.0), 1.0)
        );

        println!(
            "{:?}",
            cornell_green.convert_to_xyz(Bounds1D::new(400.0, 700.0), 1.0)
        );
    }
    #[test]
    fn test_parse_linear_spectra() {
        let spectra = load_linear("data/curves/spectra/xenon_lamp.spectra", |x| 10.0 * x).unwrap();
        println!("{}", spectra.evaluate_power(500.0));
        println!("{}", spectra.evaluate_power(500.5));
    }
}
