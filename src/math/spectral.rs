use packed_simd::f32x4;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

use crate::math::color::XYZColor;
use crate::math::misc::{gaussian, w};
use crate::math::*;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct SingleEnergy(pub f32);

impl SingleEnergy {
    pub fn new(energy: f32) -> Self {
        SingleEnergy { 0: energy }
    }
    pub const ZERO: SingleEnergy = SingleEnergy { 0: 0.0 };
    pub const ONE: SingleEnergy = SingleEnergy { 0: 1.0 };
}
impl Add for SingleEnergy {
    type Output = SingleEnergy;
    fn add(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self.0 + rhs.0)
    }
}
impl AddAssign for SingleEnergy {
    fn add_assign(&mut self, rhs: SingleEnergy) {
        self.0 += rhs.0;
    }
}

impl Mul<f32> for SingleEnergy {
    type Output = SingleEnergy;
    fn mul(self, rhs: f32) -> Self::Output {
        SingleEnergy::new(self.0 * rhs)
    }
}
impl Mul<SingleEnergy> for f32 {
    type Output = SingleEnergy;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self * rhs.0)
    }
}

impl Mul for SingleEnergy {
    type Output = SingleEnergy;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        SingleEnergy::new(self.0 * rhs.0)
    }
}

impl MulAssign for SingleEnergy {
    fn mul_assign(&mut self, other: SingleEnergy) {
        self.0 = self.0 * other.0
    }
}
impl Div<f32> for SingleEnergy {
    type Output = SingleEnergy;
    fn div(self, rhs: f32) -> Self::Output {
        SingleEnergy::new(self.0 / rhs)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SingleWavelength {
    pub lambda: f32,
    pub energy: SingleEnergy,
}

impl SingleWavelength {
    pub const fn new(lambda: f32, energy: SingleEnergy) -> SingleWavelength {
        SingleWavelength { lambda, energy }
    }

    pub fn new_from_range(x: f32, lower: f32, upper: f32) -> Self {
        SingleWavelength::new(lower + x * (upper - lower), SingleEnergy::ZERO)
    }

    pub fn with_energy(&self, energy: SingleEnergy) -> Self {
        SingleWavelength::new(self.lambda, energy)
    }
}

impl Mul<f32> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, other: f32) -> SingleWavelength {
        self.with_energy(self.energy * other)
    }
}

impl Mul<SingleWavelength> for f32 {
    type Output = SingleWavelength;
    fn mul(self, other: SingleWavelength) -> SingleWavelength {
        other.with_energy(self * other.energy)
    }
}

impl Mul<XYZColor> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, xyz: XYZColor) -> SingleWavelength {
        // let lambda = other.wavelength;
        // let other_as_color: XYZColor = other.into();
        // other_as_color gives us the x y and z values for other
        self.with_energy(self.energy * xyz.y())
    }
}

impl Div<f32> for SingleWavelength {
    type Output = SingleWavelength;
    fn div(self, other: f32) -> SingleWavelength {
        self.with_energy(self.energy / other)
    }
}

impl DivAssign<f32> for SingleWavelength {
    fn div_assign(&mut self, other: f32) {
        self.energy = self.energy / other;
    }
}

impl Mul<SingleEnergy> for SingleWavelength {
    type Output = SingleWavelength;
    fn mul(self, rhs: SingleEnergy) -> Self::Output {
        self.with_energy(self.energy * rhs)
    }
}

impl Mul<SingleWavelength> for SingleEnergy {
    type Output = SingleWavelength;
    fn mul(self, rhs: SingleWavelength) -> Self::Output {
        rhs.with_energy(self * rhs.energy)
    }
}

pub fn x_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.056, 5998.0, 379.0, 310.0)
        + gaussian(angstroms.into(), 0.362, 4420.0, 160.0, 267.0)
        + gaussian(angstroms.into(), -0.065, 5011.0, 204.0, 262.0)) as f32
}

pub fn y_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 0.821, 5688.0, 469.0, 405.0)
        + gaussian(angstroms.into(), 0.286, 5309.0, 163.0, 311.0)) as f32
}

pub fn z_bar(angstroms: f32) -> f32 {
    (gaussian(angstroms.into(), 1.217, 4370.0, 118.0, 360.0)
        + gaussian(angstroms.into(), 0.681, 4590.0, 260.0, 138.0)) as f32
}

impl From<SingleWavelength> for XYZColor {
    fn from(swss: SingleWavelength) -> Self {
        // convert to Angstroms. 10 Angstroms == 1nm
        let angstroms = swss.lambda * 10.0;

        // i don't know how to take Energy into account for this
        XYZColor::new(
            swss.energy.0 * x_bar(angstroms),
            swss.energy.0 * y_bar(angstroms),
            swss.energy.0 * z_bar(angstroms),
        )
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Op {
    Add,
    Mul,
}

#[derive(Debug, Clone)]
pub enum SPD {
    Linear {
        signal: Vec<f32>,
        bounds: Bounds1D,
    },
    Polynomial {
        xoffset: f32,
        coefficients: [f32; 8],
    },
    Cauchy {
        a: f32,
        b: f32,
    },
    Exponential {
        signal: Vec<(f32, f32, f32)>,
    },
    InverseExponential {
        signal: Vec<(f32, f32, f32)>,
    },
    Blackbody {
        temperature: f32,
        boost: f32,
    },
    Machine {
        seed: f32,
        list: Vec<(Op, SPD)>,
    },
}

pub trait SpectralPowerDistributionFunction {
    fn evaluate(&self, lambda: f32) -> f32;
    fn evaluate_power(&self, lambda: f32) -> f32;
    fn convert_to_xyz(&self, integration_bounds: Bounds1D, step_size: f32) -> XYZColor {
        let iterations =
            ((integration_bounds.upper - integration_bounds.lower) / step_size) as usize;
        let mut sum: XYZColor = XYZColor::ZERO;
        for i in 0..iterations {
            let lambda = (integration_bounds.lower + (i as f32) * step_size);
            let angstroms = lambda * 10.0;
            let val = self.evaluate_power(lambda);
            sum.0 += f32x4::new(
                val * x_bar(angstroms),
                val * y_bar(angstroms),
                val * z_bar(angstroms),
                0.0,
            ) * step_size;
        }
        sum
    }
}

impl SpectralPowerDistributionFunction for SPD {
    fn evaluate_power(&self, lambda: f32) -> f32 {
        match (&self) {
            SPD::Linear { signal, bounds } => {
                assert!(
                    bounds.lower <= lambda && lambda < bounds.upper,
                    "lambda was {:?}, bounds were {:?}",
                    lambda,
                    bounds
                );
                let step_size = (bounds.upper - bounds.lower) / (signal.len() as f32);
                let index = ((lambda - bounds.lower) / step_size) as usize;
                signal[index]
            }
            SPD::Polynomial {
                xoffset,
                coefficients,
            } => {
                let mut val = 0.0;
                let tmp_lambda = lambda - xoffset;
                for (i, &coef) in coefficients.iter().enumerate() {
                    val += coef * tmp_lambda.powi(i as i32);
                }
                val
            }
            SPD::Cauchy { a, b } => *a + *b / (lambda * lambda),
            SPD::Exponential { signal } => {
                let mut val = 0.0f32;
                for &(o, s, m) in signal {
                    val += w(lambda, m, o, s);
                }
                val
            }
            SPD::InverseExponential { signal } => {
                let mut val = 1.0f32;
                for &(o, s, m) in signal {
                    val -= w(lambda, m, o, s);
                }
                val.max(0.0)
            }
            SPD::Machine { seed, list } => {
                let mut val = *seed;
                for (op, spd) in list {
                    let eval = spd.evaluate_power(lambda);
                    val = match (op) {
                        Op::Add => val + eval,
                        Op::Mul => val * eval,
                    };
                }
                val.max(0.0)
            }
            SPD::Blackbody { temperature, boost } => {
                if *boost == 0.0 {
                    blackbody(*temperature, lambda)
                } else {
                    boost * blackbody(*temperature, lambda)
                        / blackbody(*temperature, max_blackbody_lambda(*temperature))
                }
            }
        }
    }
    fn evaluate(&self, lambda: f32) -> f32 {
        self.evaluate_power(lambda).min(1.0)
    }
}

pub trait RereflectanceFunction {
    fn evaluate(&self, lambda: f32, energy: f32) -> (f32, f32);
}
