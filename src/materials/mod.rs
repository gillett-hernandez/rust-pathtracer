// pub use crate::material::{BRDF, BTDF, BXDF, PDF};
pub use crate::material::{BRDF, PDF};
use crate::math::*;

pub mod lambertian;
pub use lambertian::Lambertian;

pub type MatTableId = u8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lambertian() {
        let lambertian = Lambertian::new(RGBColor::new(0.9, 0.2, 0.9));
        // simulate incoming ray from directly above
        let incoming: Ray = Ray::new(Point3::new(0.0, 0.0, 10.0), -Vec3::Z);
        let v = lambertian.generate(Sample2D::new_random_sample(), incoming.direction);
        println!("{:?}", lambertian.f(incoming.direction, v));
        assert!(lambertian.value(incoming.direction, v) > 0.0);
        println!("{:?}", v);
        assert!(v * Vec3::Z > 0.0);
    }
}
