// use std::vec::Vec;
#[derive(Clone)]
pub struct Film<T> {
    pub buffer: Vec<T>,
    pub width: usize,
    pub height: usize,
}

impl<T: Copy> Film<T> {
    pub fn new(width: usize, height: usize, fill_value: T) -> Film<T> {
        // allocate with
        let capacity: usize = (width * height) as usize;
        let mut buffer: Vec<T> = Vec::with_capacity(capacity as usize);
        for _ in 0..capacity {
            buffer.push(fill_value);
        }
        Film {
            buffer,
            width,
            height,
        }
    }
    pub fn at(&self, x: usize, y: usize) -> T {
        self.buffer[y * self.width + x]
    }

    pub fn at_uv(&self, mut uv: (f32, f32)) -> T {
        // debug_assert!(uv.0 < 1.0 && uv.1 < 1.0 && uv.0 >= 0.0 && uv.1 >= 0.0);
        uv.0 = uv.0.clamp(0.0, 0.999999);
        uv.1 = uv.1.clamp(0.0, 0.999999);
        self.at(
            (uv.0 * (self.width as f32)) as usize,
            (uv.1 * (self.height as f32)) as usize,
        )
    }

    // pub fn at_mut(&mut self, x: usize, y: usize) -> &mut T {
    //     &mut self.buffer[y * self.width + x]
    // }

    pub fn write_at(&mut self, x: usize, y: usize, value: T) {
        self.buffer[y * self.width + x] = value
    }

    pub fn total_pixels(&self) -> usize {
        self.width * self.height
    }
}
