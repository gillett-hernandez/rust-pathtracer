#[derive(Copy, Clone, Debug)]
pub struct UV(pub f32, pub f32);

impl From<(f32, f32)> for UV {
    fn from(value: (f32, f32)) -> Self {
        UV(value.0, value.1)
    }
}

impl From<UV> for (f32, f32) {
    fn from(uv: UV) -> Self {
        (uv.0, uv.1)
    }
}

#[derive(Clone, Debug)]
pub struct Vec2D<T> {
    pub buffer: Vec<T>,
    pub width: usize,
    pub height: usize,
}

impl<T: Copy> Vec2D<T> {
    pub fn new(width: usize, height: usize, fill_value: T) -> Vec2D<T> {
        Vec2D {
            buffer: vec![fill_value; width * height],
            width,
            height,
        }
    }
    pub fn at(&self, x: usize, y: usize) -> T {
        self.buffer[y * self.width + x]
    }
    pub fn at_uv(&self, mut uv: UV) -> T {
        // debug_assert!(uv.0 < 1.0 && uv.1 < 1.0 && uv.0 >= 0.0 && uv.1 >= 0.0);
        uv.0 = uv.0.clamp(0.0, 1.0 - f32::EPSILON);
        uv.1 = uv.1.clamp(0.0, 1.0 - f32::EPSILON);
        self.at(
            (uv.0 * (self.width as f32)) as usize,
            (uv.1 * (self.height as f32)) as usize,
        )
    }
}

impl<T> Vec2D<T> {
    pub fn write_at(&mut self, x: usize, y: usize, value: T) {
        self.buffer[y * self.width + x] = value
    }

    pub fn total_pixels(&self) -> usize {
        self.width * self.height
    }
}
