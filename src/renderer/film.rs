// use std::vec::Vec;
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

    pub fn total_pixels(&self) -> usize {
        self.width * self.height
    }
}
