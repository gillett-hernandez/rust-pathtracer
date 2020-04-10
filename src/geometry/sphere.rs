use crate::hittable::{HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

/*
class sphere : public hittable
{
public:
    sphere() {}

    sphere(vec3 cen, float r, material *m)
        : center(cen), radius(r), mat_ptr(m){};

    virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    virtual bool bounding_box(float t0, float t1, aabb &box) const;
    virtual float pdf_value(const vec3 &o, const vec3 &v) const
    {
        hit_record rec;
        if (this->hit(ray(o, v), 0.001, FLT_MAX, rec))
        {
            float cos_theta_max = sqrt(1 - radius * radius / (center - o).squared_length());
            float solid_angle = 2 * M_PI * (1 - cos_theta_max);
            return 1 / solid_angle;
        }
        else
        {
            return 0;
        }
    }
    virtual vec3 random(const vec3 &o) const
    {
        vec3 direction = center - o;
        float distance_squared = direction.squared_length();
        onb uvw;
        uvw.build_from_w(direction);
        return uvw.local(random_to_sphere(radius, distance_squared));
    }
    vec3 center;
    float radius;
    material *mat_ptr;
};


bool sphere::bounding_box(float t0, float t1, aabb &box) const
{
    box = aabb(center - vec3(radius, radius, radius),
               center + vec3(radius, radius, radius));
    return true;
}*/

pub struct Sphere {
    pub radius: f32,
    pub origin: Point3,
    pub material_id: Option<MaterialId>,
}

impl Sphere {
    pub fn new(radius: f32, origin: Point3, material_id: Option<MaterialId>) -> Sphere {
        Sphere {
            radius,
            origin,
            material_id,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let oc: Vec3 = r.origin - self.origin;
        let a = r.direction * r.direction;
        let b = oc * r.direction;
        let c = oc * oc - self.radius * self.radius;
        let discriminant = b * b - a * c;
        let discriminant_sqrt = discriminant.sqrt();
        if discriminant > 0.0 {
            let mut time: f32;
            let point: Point3;
            let normal: Vec3;
            time = (-b - discriminant_sqrt) / a;
            if time < t1 && time > t0 {
                point = r.point_at_parameter(time);
                normal = (point - self.origin) / self.radius;
                //         rec.mat_ptr = mat_ptr;
                //         rec.primitive = (hittable *)this;
                return Some(HitRecord {
                    time,
                    point,
                    normal,
                    material: self.material_id,
                });
            }
            time = (-b + discriminant_sqrt) / a;
            if time < t1 && time > t0 {
                point = r.point_at_parameter(time);
                normal = (point - self.origin) / self.radius;
                //         rec.mat_ptr = mat_ptr;
                //         rec.primitive = (hittable *)this;
                return Some(HitRecord {
                    time,
                    point,
                    normal,
                    material: self.material_id,
                });
            }
        }
        None
    }
}
