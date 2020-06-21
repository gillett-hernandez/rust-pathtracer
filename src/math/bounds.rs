#[derive(Copy, Clone, Debug)]
pub struct Bounds1D {
    pub lower: f32,
    pub upper: f32,
}

impl Bounds1D {
    pub const fn new(lower: f32, upper: f32) -> Self {
        Bounds1D { lower, upper }
    }
}
