use super::random;
use rand::seq::SliceRandom;
use rand::{thread_rng, RngCore};

use std::f32::EPSILON;

#[derive(Debug)]
pub struct Sample1D {
    pub x: f32,
}

impl Sample1D {
    pub fn new(x: f32) -> Self {
        debug_assert!(x < 1.0 && x >= 0.0);
        Sample1D { x }
    }
    pub fn new_random_sample() -> Self {
        Sample1D::new(random())
    }
    pub fn choose<T>(mut self, split: f32, a: T, b: T) -> (Self, T) {
        debug_assert!(0.0 <= split && split <= 1.0);
        debug_assert!(self.x >= 0.0 && self.x < 1.0);
        if self.x < split {
            assert!(split > 0.0);
            self.x /= split;
            (self, a)
        } else {
            // if split was 1.0, there's no way for self.x to be greather than or equal to it
            // since self.x in [0, 1)
            debug_assert!(split < 1.0);
            self.x = (self.x - split) / (1.0 - split);
            (self, b)
        }
    }
}

#[derive(Debug)]
pub struct Sample2D {
    pub x: f32,
    pub y: f32,
}

impl Sample2D {
    pub fn new(x: f32, y: f32) -> Self {
        debug_assert!(x < 1.0 && x >= 0.0);
        debug_assert!(y < 1.0 && y >= 0.0);

        Sample2D { x, y }
    }
    pub fn new_random_sample() -> Self {
        Sample2D::new(random(), random())
    }
}
#[derive(Debug)]
pub struct Sample3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Sample3D {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Sample3D { x, y, z }
    }
    pub fn new_random_sample() -> Self {
        Sample3D::new(random(), random(), random())
    }
}

pub trait Sampler {
    fn draw_1d(&mut self) -> Sample1D;
    fn draw_2d(&mut self) -> Sample2D;
    fn draw_3d(&mut self) -> Sample3D;
}

pub struct RandomSampler {}

impl RandomSampler {
    pub const fn new() -> RandomSampler {
        RandomSampler {}
    }
}

impl Sampler for RandomSampler {
    fn draw_1d(&mut self) -> Sample1D {
        Sample1D::new_random_sample()
    }
    fn draw_2d(&mut self) -> Sample2D {
        Sample2D::new_random_sample()
    }
    fn draw_3d(&mut self) -> Sample3D {
        Sample3D::new_random_sample()
    }
}
pub struct StratifiedSampler {
    pub dims: [usize; 3],
    pub indices: [usize; 3],
    pub first: Vec<usize>,
    pub second: Vec<usize>,
    pub third: Vec<usize>,
    rng: Box<dyn RngCore>,
}

impl StratifiedSampler {
    pub fn new(xdim: usize, ydim: usize, zdim: usize) -> Self {
        StratifiedSampler {
            dims: [xdim, ydim, zdim],
            indices: [0, 0, 0],
            first: (0..xdim).into_iter().collect(),
            second: (0..(xdim * ydim)).into_iter().collect(),
            third: (0..(xdim * ydim * zdim)).into_iter().collect(),
            rng: Box::new(thread_rng()),
        }
    }
}

impl Sampler for StratifiedSampler {
    fn draw_1d(&mut self) -> Sample1D {
        if self.indices[0] == 0 {
            // shuffle, then draw.
            self.first.shuffle(&mut self.rng);
            // print!("#");
        }
        let idx = self.first[self.indices[0]];
        let (width, _depth, _height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[0] += 1;
        if self.indices[0] >= width {
            self.indices[0] = 0;
        }
        // convert idx to the "pixel" based on dims
        let mut sample = Sample1D::new_random_sample();
        let x = idx;
        let old_x = sample.x;
        sample.x = (sample.x + x as f32) / (width as f32);
        if sample.x == 1.0 {
            sample.x -= EPSILON;
        }
        debug_assert!(
            sample.x < 1.0 && sample.x >= 0.0,
            "{:?} = ({:?} + {:?})/{:?}",
            sample.x,
            old_x,
            x,
            width,
        );
        sample
    }
    fn draw_2d(&mut self) -> Sample2D {
        if self.indices[1] == 0 {
            // shuffle, then draw.
            self.second.shuffle(&mut self.rng);
            // print!("#");
        }
        let idx = self.second[self.indices[1]];
        let (width, depth, _height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[1] += 1;
        if self.indices[1] >= width * depth {
            self.indices[1] = 0;
        }
        // convert idx to the "pixel" based on dims
        let (x, y) = (idx % width, idx / width);
        let mut sample = Sample2D::new_random_sample();
        let old_x = sample.x;
        sample.x = (sample.x + x as f32) / (width as f32);
        let old_y = sample.y;
        sample.y = (sample.y + y as f32) / (depth as f32);
        if sample.x == 1.0 {
            sample.x -= EPSILON;
        }
        if sample.y == 1.0 {
            sample.y -= EPSILON;
        }
        debug_assert!(
            sample.x < 1.0 && sample.x >= 0.0,
            "{:?} = ({:?} + {:?})/{:?}",
            sample.x,
            old_x,
            x,
            width,
        );
        debug_assert!(
            sample.y < 1.0 && sample.y >= 0.0,
            "{:?} = ({:?} + {:?})/{:?}",
            sample.y,
            old_y,
            y,
            depth,
        );
        sample
    }
    fn draw_3d(&mut self) -> Sample3D {
        if self.indices[2] == 0 {
            // shuffle, then draw.
            self.third.shuffle(&mut self.rng);
            // print!("#");
        }
        let idx = self.third[self.indices[2]];
        let (width, depth, height) = (self.dims[0], self.dims[1], self.dims[2]);
        self.indices[2] += 1;
        if self.indices[2] >= width * depth * height {
            self.indices[2] = 0;
        }
        // idx = x + width * y + width * depth * z
        // convert idx to the "pixel" based on dims
        // z coordinate is how many slices high the sample is
        let z = idx / (depth * width);
        // y coordinate is how far into a slice a given "pixel" is
        let y = (idx / width) % depth;
        // x coordinate is how far along width a given pixel is
        let x = idx % width;
        let mut sample = Sample3D::new_random_sample();
        sample.x = (sample.x + x as f32) / (width as f32);
        sample.y = (sample.y + y as f32) / (depth as f32);
        sample.z = (sample.z + z as f32) / (height as f32);
        if sample.x == 1.0 {
            sample.x -= EPSILON;
        }

        if sample.y == 1.0 {
            sample.y -= EPSILON;
        }
        if sample.z == 1.0 {
            sample.z -= EPSILON;
        }
        debug_assert!(sample.x < 1.0 && sample.x >= 0.0);
        debug_assert!(sample.y < 1.0 && sample.y >= 0.0);
        debug_assert!(sample.z < 1.0 && sample.z >= 0.0);
        sample
    }
}

#[cfg(test)]
mod test {
    use super::*;
    fn function(x: f32) -> f32 {
        x * x - x + 1.0
    }
    #[test]
    fn test_random_sampler_1d() {
        let mut sampler = Box::new(RandomSampler::new());
        let mut s = 0.0;
        for _i in 0..10000000 {
            let sample = sampler.draw_1d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            s += function(sample.x);
        }
        println!("{}", s / 10000000.0);
    }
    #[test]
    fn test_stratified_sampler_1d() {
        let mut sampler = Box::new(StratifiedSampler::new(20, 20, 10));
        let mut s = 0.0;
        for _i in 0..10000000 {
            let sample = sampler.draw_1d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            s += function(sample.x);
        }
        println!("{}", s / 10000000.0);
    }
    #[test]
    fn test_stratified_sampler_2d() {
        let mut sampler = Box::new(StratifiedSampler::new(20, 20, 10));
        for _ in 0..10000 {
            sampler.draw_1d();
        }
        for _i in 0..10000000 {
            let sample = sampler.draw_2d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y < 1.0, "{}", sample.y);
        }
    }
    #[test]
    fn test_stratified_sampler_3d() {
        let mut sampler = Box::new(StratifiedSampler::new(20, 20, 10));

        for _i in 0..10000000 {
            let sample = sampler.draw_3d();
            assert!(0.0 <= sample.x && sample.x < 1.0, "{}", sample.x);
            assert!(0.0 <= sample.y && sample.y < 1.0, "{}", sample.y);
            assert!(0.0 <= sample.z && sample.z < 1.0, "{}", sample.z);
        }
    }
}
