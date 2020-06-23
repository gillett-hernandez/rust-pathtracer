use crate::curves::VISIBLE_RANGE;
use crate::math::*;
use crate::spectral::InterpolationMode;

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub fn parse_tabulated_curve_from_csv<F>(
    filename: &str,
    column: usize,
    interpolation_mode: InterpolationMode,
    mut func: F,
) -> Result<SPD, Box<dyn Error>>
where
    F: Fn(f32) -> f32,
{
    let path = Path::new(filename);
    let mut file = File::open(path)?;
    let mut signal: Vec<(f32, f32)> = Vec::new();
    let mut buf = String::new();
    file.read_to_string(&mut buf);
    for line in buf.split_terminator("\n") {
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
                    println!("skipped csv line {:?} {:?} from file {}", a, b, filename);
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_tabulated_curve() {
        let gold_ior_tabulation = parse_tabulated_curve_from_csv(
            "data/curves/gold.csv",
            1,
            InterpolationMode::Cubic,
            |x: f32| x * 1000.0,
        )
        .unwrap();
        let gold_kappa_tabulation = parse_tabulated_curve_from_csv(
            "data/curves/gold.csv",
            2,
            InterpolationMode::Cubic,
            |x: f32| x * 1000.0,
        )
        .unwrap();

        println!("{:?}", gold_ior_tabulation.evaluate_power(500.0));
        println!("{:?}", gold_kappa_tabulation.evaluate_power(500.0));
    }
}
