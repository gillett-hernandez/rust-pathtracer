pub type Float = f32;

pub struct Vec3 {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Vec3 {
    pub const fn new(x: Float, y: Float, z: Float) -> Vec3 {
        Vec3 { x, y, z }
    }
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);
}

pub struct RGBColor {
    pub r: Float,
    pub g: Float,
    pub b: Float,
}

impl RGBColor {
    pub const fn new(r: Float, g: Float, b: Float) -> RGBColor {
        RGBColor { r, g, b }
    }
    pub const ZERO: RGBColor = RGBColor::new(0.0, 0.0, 0.0);
}

pub struct Point2 {
    pub x: Float,
    pub y: Float,
}

pub struct Point3 {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
}
