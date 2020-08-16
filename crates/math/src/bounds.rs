#[derive(Copy, Clone, Debug)]
pub struct Bounds1D {
    pub lower: f32,
    pub upper: f32,
}

impl Bounds1D {
    pub const fn new(lower: f32, upper: f32) -> Self {
        Bounds1D { lower, upper }
    }
    pub fn span(&self) -> f32 {
        self.upper - self.lower
    }

    pub fn contains(&self, value: &f32) -> bool {
        &self.lower <= value && value < &self.upper
    }
    pub fn intersection(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.max(other.lower), self.upper.min(other.upper))
    }

    pub fn union(&self, other: Self) -> Self {
        Bounds1D::new(self.lower.min(other.lower), self.upper.max(other.upper))
    }
}
