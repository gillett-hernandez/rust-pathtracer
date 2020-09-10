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

#[derive(Copy, Clone, Debug)]
pub struct Bounds2D {
    pub x: Bounds1D,
    pub y: Bounds1D,
}

impl Bounds2D {
    pub const fn new(x: Bounds1D, y: Bounds1D) -> Self {
        Bounds2D { x, y }
    }
    pub fn area(&self) -> f32 {
        self.x.span() * self.y.span()
    }

    pub fn contains(&self, value: (f32, f32)) -> bool {
        self.x.contains(&value.0) && self.y.contains(&value.1)
    }
    pub fn intersection(&self, other: Self) -> Self {
        Bounds2D::new(self.x.intersection(other.x), self.y.intersection(other.y))
    }

    pub fn union(&self, other: Self) -> Self {
        Bounds2D::new(self.x.union(other.x), self.y.union(other.y))
    }
}
