use crate::hittable::{HasBoundingBox, HitRecord, Hittable, AABB};
use crate::materials::MaterialId;
use crate::math::*;

use crate::accelerator::BHShape;
use crate::geometry::*;

use std::cmp::{Ordering, PartialOrd};

#[derive(Clone, Debug)]
pub struct Instance {
    pub aggregate: Aggregate,
    pub transform: Option<Transform3>,
    pub material_id: Option<MaterialId>,
    pub instance_id: usize,
    node_id: usize,
}

impl PartialEq for Instance {
    fn eq(&self, other: &Instance) -> bool {
        self.instance_id == other.instance_id
    }
}

impl Eq for Instance {}

impl PartialOrd for Instance {
    fn partial_cmp(&self, other: &Instance) -> Option<Ordering> {
        if self.instance_id < other.instance_id {
            Some(Ordering::Less)
        } else if self.instance_id > other.instance_id {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl Ord for Instance {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Instance {
    pub fn new(
        aggregate: Aggregate,
        transform: Option<Transform3>,
        material_id: Option<MaterialId>,
        instance_id: usize,
    ) -> Self {
        Instance {
            aggregate,
            transform,
            material_id,
            instance_id,
            node_id: 0,
        }
    }

    // fn with_transform(&mut self, transform: Transform3) {
    //     // replaces this instance's transform with a new one
    //     self.transform = Some(transform);
    // }
}
impl HasBoundingBox for Instance {
    fn aabb(&self) -> AABB {
        let mut aabb = self.aggregate.aabb();
        if let Some(transform) = self.transform {
            aabb = transform.to_world(aabb)
        }
        aabb
    }
}

impl Hittable for Instance {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        debug_assert!(r.origin.is_finite());
        debug_assert!(r.direction.is_finite());
        debug_assert!(
            t0.is_finite(),
            "{:?} {:?} {:?} {:?} {:?} {:?}",
            self.transform,
            self.instance_id,
            self.material_id,
            r,
            t0,
            t1
        );
        debug_assert!(!t1.is_nan());
        if let Some(transform) = self.transform {
            if let Some(hit) = self.aggregate.hit(
                Ray {
                    origin: transform.to_local(r.origin),
                    direction: transform.to_local(r.direction),
                    ..r
                },
                t0,
                t1,
            ) {
                debug_assert!(
                    hit.point.is_finite() && hit.normal.is_finite() && hit.time.is_finite(),
                    "{:?}",
                    hit
                );
                debug_assert!(hit.uv.0 <= 1.0 && hit.uv.1 <= 1.0, "{:?}", hit);
                debug_assert!(hit.uv.0 >= 0.0 && hit.uv.1 >= 0.0, "{:?}", hit);
                Some(HitRecord {
                    normal: transform.to_world(hit.normal).normalized(),
                    point: transform.to_world(hit.point),
                    instance_id: self.instance_id,
                    material: self.material_id.unwrap_or(hit.material),
                    ..hit
                })
            } else {
                None
            }
        } else {
            if let Some(hit) = self.aggregate.hit(r, t0, t1) {
                debug_assert!(
                    hit.point.is_finite() && hit.normal.is_finite() && hit.time.is_finite(),
                    "{:?}",
                    hit
                );
                debug_assert!(hit.uv.0 <= 1.0 && hit.uv.1 <= 1.0, "{:?}", hit);
                debug_assert!(hit.uv.0 >= 0.0 && hit.uv.1 >= 0.0, "{:?}", hit);
                Some(HitRecord {
                    instance_id: self.instance_id,
                    material: self.material_id.unwrap_or(hit.material),
                    ..hit
                })
            } else {
                None
            }
        }
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        if let Some(transform) = self.transform {
            let (vec, pdf) = self.aggregate.sample(s, transform.to_local(from));
            (transform.to_world(vec).normalized(), pdf)
        } else {
            self.aggregate.sample(s, from)
        }
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        if let Some(transform) = self.transform {
            let (point, normal, pdf) = self.aggregate.sample_surface(s);
            (
                transform.to_world(point),
                transform.to_world(normal).normalized(),
                pdf,
            )
        } else {
            self.aggregate.sample_surface(s)
        }
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        let (normal, from, to) = if let Some(transform) = self.transform {
            (
                transform.to_world(normal).normalized(),
                transform.to_world(from),
                transform.to_world(to),
            )
        } else {
            (normal, from, to)
        };
        self.aggregate.pdf(normal, from, to)
    }

    fn surface_area(&self, transform: &Transform3) -> f32 {
        if let Some(more_transform) = self.transform {
            self.aggregate
                .surface_area(&(more_transform * (*transform)))
        } else {
            self.aggregate.surface_area(transform)
        }
    }
}

impl Instance {
    pub fn get_instance_id(&self) -> usize {
        self.instance_id
    }
    pub fn get_material_id(&self) -> MaterialId {
        self.material_id.unwrap_or(0u16.into())
    }
}

impl BHShape for Instance {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_id = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_id
    }
}

// impl From<Aggregate> for Instance {
//     fn from(data: Aggregate) -> Self {
//         Instance::new(data, None, Some(material_id), Some(instance_id))
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_instance() {
        let sphere = Sphere::new(2.0, Point3::ORIGIN);
        let aarect = AARect::new((4.0, 4.0), Point3::ORIGIN, Axis::Z, true);

        let transform = Transform3::from_stack(
            Some(Transform3::from_scale(Vec3::new(3.0, 3.0, 3.0))),
            Some(Transform3::from_axis_angle(Vec3::Z, 1.0)),
            Some(Transform3::from_translation(Vec3::new(1.0, 1.0, 1.0))),
        );

        let aggregate1 = Aggregate::from(sphere);
        let aggregate2 = Aggregate::from(aarect);

        let instance1 = Instance::new(aggregate1, Some(transform), Some(0.into()), 0);
        let instance2 = Instance::new(aggregate2, Some(transform), Some(0.into()), 0);

        let test_ray = Ray::new(Point3::ORIGIN + 10.0 * Vec3::Z, -Vec3::Z);

        let isect1 = instance1.hit(test_ray, 0.0, INFINITY);
        let isect2 = instance2.hit(test_ray, 0.0, INFINITY);

        println!("ray was {:?}", test_ray);
        println!(
            "in local space should be {:?}",
            transform.to_local(test_ray)
        );
        println!(
            "in world space should be {:?}",
            transform.to_world(test_ray)
        );
        if let Some(hit) = isect1 {
            println!("{:?}", hit);
        }

        if let Some(hit) = isect2 {
            println!("{:?}", hit);
        }
    }
}
