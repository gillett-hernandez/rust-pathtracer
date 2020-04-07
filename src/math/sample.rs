use crate::math::*;

#[derive(Debug)]
pub struct Sample1D {
    pub x: f32,
}

#[derive(Debug)]
pub struct Sample2D {
    pub x: f32,
    pub y: f32,
}

impl Sample2D {
    pub fn new_random_sample() -> Self {
        Sample2D {
            x: random(),
            y: random(),
        }
    }
}
#[derive(Debug)]
pub struct Sample3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
