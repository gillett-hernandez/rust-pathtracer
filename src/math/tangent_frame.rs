use crate::math::*;

// also known as an orthonormal basis.
pub struct TangentFrame {
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub normal: Vec3,
}

impl TangentFrame {
    pub fn new(tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Self {
        assert!(
            (tangent * bitangent).abs() < 0.000001,
            "tbit:{:?} * {:?} was != 0",
            tangent,
            bitangent
        );
        assert!(
            (tangent * normal).abs() < 0.000001,
            "tn: {:?} * {:?} was != 0",
            tangent,
            normal
        );
        assert!(
            (bitangent * normal).abs() < 0.000001,
            "bitn:{:?} * {:?} was != 0",
            bitangent,
            normal
        );
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: bitangent.normalized(),
            normal: normal.normalized(),
        }
    }
    pub fn from_tangent_and_normal(tangent: Vec3, normal: Vec3) -> Self {
        TangentFrame {
            tangent: tangent.normalized(),
            bitangent: tangent.normalized().cross(normal.normalized()).normalized(),
            normal: normal.normalized(),
        }
    }

    pub fn from_normal(normal: Vec3) -> Self {
        // let n2 = Vec3::from_raw(normal.0 * normal.0);
        // let (x, y, z) = (normal.x(), normal.y(), normal.z());
        let [x, y, z, _]: [f32; 4] = normal.0.into();
        let sign = (1.0 as f32).copysign(z);
        let a = -1.0 / (sign + z);
        let b = x * y * a;
        TangentFrame {
            tangent: Vec3::new(1.0 + sign * x * x * a, sign * b, -sign * x),
            bitangent: Vec3::new(b, sign + y * y * a, -y),
            normal,
        }
    }

    #[inline(always)]
    pub fn to_world(&self, v: &Vec3) -> Vec3 {
        self.tangent * v.x() + self.bitangent * v.y() + self.normal * v.z()
    }

    #[inline(always)]
    pub fn to_local(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
            self.tangent * (*v),
            self.bitangent * (*v),
            self.normal * (*v),
        )
    }
}
