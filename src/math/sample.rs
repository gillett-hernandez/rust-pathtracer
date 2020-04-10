use crate::math::*;

#[derive(Debug)]
pub struct Sample1D {
    pub x: f32,
}

impl Sample1D {
    pub fn new_random_sample() -> Self {
        Sample1D { x: random() }
    }
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

impl Sample3D {
    pub fn new_random_sample() -> Self {
        Sample3D {
            x: random(),
            y: random(),
            z: random(),
        }
    }
}

pub trait Sampler {
    fn draw_1d(&self) -> Sample1D;
    fn draw_2d(&self) -> Sample2D;
    fn draw_3d(&self) -> Sample3D;
}

pub struct RandomSampler {}

impl RandomSampler {
    pub const fn new() -> RandomSampler {
        RandomSampler {}
    }
}

impl Sampler for RandomSampler {
    fn draw_1d(&self) -> Sample1D {
        Sample1D::new_random_sample()
    }
    fn draw_2d(&self) -> Sample2D {
        Sample2D::new_random_sample()
    }
    fn draw_3d(&self) -> Sample3D {
        Sample3D::new_random_sample()
    }
}
