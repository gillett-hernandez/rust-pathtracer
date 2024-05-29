pub use rayon::prelude::*;
pub use std::simd::{
    cmp::{SimdPartialEq, SimdPartialOrd},
    f32x4,
    num::{SimdFloat, SimdInt},
    simd_swizzle, StdFloat,
};

pub use crate::camera::{Camera, CameraEnum, CameraId};
pub use crate::curves::*;
pub use crate::geometry::InstanceId;
pub use crate::materials::{Material, MaterialEnum, MaterialId, MediumId};
pub use crate::texture::TexStack;
pub use crate::tonemap::{Color, Tonemapper};
pub use crate::vec2d::{Vec2D, UV};
pub use crate::{rgb_to_u32, TransportMode, INTERSECTION_TIME_OFFSET, MAUVE, NORMAL_OFFSET};
pub use lazy_static::lazy_static;

#[cfg(feature = "preview")]
pub use crate::{update_window_buffer, window_loop};

pub use math::prelude::*;
pub use math::spectral::{BOUNDED_VISIBLE_RANGE, EXTENDED_VISIBLE_RANGE};
pub use math::traits::{CheckInf, CheckNAN, CheckResult, FromScalar, ToScalar};

pub use std::cmp::Ordering;
pub use std::f32::consts::{PI, SQRT_2, TAU};

#[cfg(feature = "pbr")]
pub use pbr::ProgressBar;

// dummy ProgressBar implementation that gets used instead of the actual one, to avoid having to put #[cfg] directives around every use and method call of ProgressBar
#[cfg(not(feature = "pbr"))]
use std::io::{Stdout, Write};
#[cfg(not(feature = "pbr"))]
use std::marker::PhantomData;
#[cfg(not(feature = "pbr"))]
pub struct ProgressBar<T: Write + Send + Sync> {
    phantom: PhantomData<T>,
}

#[cfg(not(feature = "pbr"))]

impl ProgressBar<Stdout> {
    pub fn new(width: u64) -> Self {
        ProgressBar {
            phantom: PhantomData,
        }
    }
}
#[cfg(not(feature = "pbr"))]
impl<T: Write + Send + Sync> ProgressBar<T> {
    pub fn on(write: T, total: u64) -> Self {
        ProgressBar {
            phantom: PhantomData,
        }
    }
    pub fn add(&mut self, n: u64) {}
    pub fn finish(&mut self) {}
}
