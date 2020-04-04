type Float = f32;

pub struct Vec3 {
    x: Float,
    y: Float,
    z: Float,
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

pub struct Point2 {
    x: Float,
    y: Float,
}

pub struct Point3 {
    x: Float,
    y: Float,
    z: Float,
}
