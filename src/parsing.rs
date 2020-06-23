use crate::curves::VISIBLE_RANGE;
use crate::math::*;
pub use crate::spectral::InterpolationMode;

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub fn parse_tabulated_curve_from_csv<F>(
    data: &str,
    column: usize,
    interpolation_mode: InterpolationMode,
    mut func: F,
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

pub fn load_ior_and_kappa<F>(filename: &str, mut func: F) -> Result<(SPD, SPD), Box<dyn Error>>
where
    F: Clone + Copy + Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf);
    let ior = parse_tabulated_curve_from_csv(buf.as_ref(), 1, InterpolationMode::Cubic, func)?;
    let kappa = parse_tabulated_curve_from_csv(buf.as_ref(), 2, InterpolationMode::Cubic, func)?;
    Ok((ior, kappa))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_tabulated_curve() {
        let (gold_ior, gold_kappa) =
            load_ior_and_kappa("data/curves/gold.csv", |x: f32| x * 1000.0);

        println!("{:?}", gold_ior_tabulation.evaluate_power(500.0));
        println!("{:?}", gold_kappa_tabulation.evaluate_power(500.0));
    }
}
