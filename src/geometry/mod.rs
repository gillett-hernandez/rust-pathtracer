mod cube;
mod disk;
mod rect;
mod sphere;

pub use disk::Disk;
pub use rect::AARect;
pub use sphere::Sphere;

use crate::hittable::{HasBoundingBox, HitRecord, Hittable, AABB};
use crate::materials::MaterialId;
use crate::math::*;

pub enum Aggregate {
    AARect(AARect),
    Sphere(Sphere),
    Disk(Disk),
}

impl From<Sphere> for Aggregate {
    fn from(data: Sphere) -> Self {
        Aggregate::Sphere(data)
    }
}

impl From<Disk> for Aggregate {
    fn from(data: Disk) -> Self {
        Aggregate::Disk(data)
    }
}

impl From<AARect> for Aggregate {
    fn from(data: AARect) -> Self {
        Aggregate::AARect(data)
    }
}

impl HasBoundingBox for Aggregate {
    fn bounding_box(&self) -> AABB {
        match self {
            Aggregate::Sphere(sphere) => sphere.bounding_box(),
            Aggregate::AARect(rect) => rect.bounding_box(),
            Aggregate::Disk(disk) => disk.bounding_box(),
        }
    }
}

impl Hittable for Aggregate {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        match self {
            Aggregate::Sphere(sphere) => sphere.hit(r, t0, t1),
            Aggregate::Disk(disk) => disk.hit(r, t0, t1),
            Aggregate::AARect(rect) => rect.hit(r, t0, t1),
        }
    }
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, PDF) {
        match self {
            Aggregate::Sphere(sphere) => sphere.sample(s, from),
            Aggregate::Disk(disk) => disk.sample(s, from),
            Aggregate::AARect(rect) => rect.sample(s, from),
        }
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        match self {
            Aggregate::Sphere(sphere) => sphere.sample_surface(s),
            Aggregate::Disk(disk) => disk.sample_surface(s),
            Aggregate::AARect(rect) => rect.sample_surface(s),
        }
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        match self {
            Aggregate::Disk(disk) => disk.pdf(normal, from, to),
            Aggregate::Sphere(sphere) => sphere.pdf(normal, from, to),
            Aggregate::AARect(rect) => rect.pdf(normal, from, to),
        }
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        match self {
            Aggregate::Sphere(sphere) => sphere.surface_area(transform),
            Aggregate::Disk(disk) => disk.surface_area(transform),
            Aggregate::AARect(rect) => rect.surface_area(transform),
        }
    }
    fn get_instance_id(&self) -> usize {
        match self {
            Aggregate::Sphere(sphere) => sphere.get_instance_id(),
            Aggregate::Disk(disk) => disk.get_instance_id(),
            Aggregate::AARect(rect) => rect.get_instance_id(),
        }
    }
    fn get_material_id(&self) -> MaterialId {
        match self {
            Aggregate::Disk(disk) => disk.get_material_id(),
            Aggregate::Sphere(sphere) => sphere.get_material_id(),
            Aggregate::AARect(rect) => rect.get_material_id(),
        }
    }
}

pub struct Instance {
    aggregate: Aggregate,
    transform: Option<Transform3>,
    material_id: MaterialId,
    instance_id: usize,
}
impl Instance {
    fn new(
        aggregate: Aggregate,
        transform: Option<Transform3>,
        material_id: Option<MaterialId>,
        instance_id: Option<usize>,
    ) -> Self {
        // steal instance id from Aggregate if not provided.
        let instance_id = if let Some(id) = instance_id {
            id
        } else {
            aggregate.get_instance_id()
        };
        let material_id = if let Some(id) = material_id {
            id
        } else {
            aggregate.get_material_id()
        };
        Instance {
            aggregate,
            transform,
            material_id,
            instance_id,
        }
    }

    // fn with_transform(&mut self, transform: Transform3) {
    //     // replaces this instance's transform with a new one
    //     self.transform = Some(transform);
    // }
}
impl HasBoundingBox for Instance {
    fn bounding_box(&self) -> AABB {
        let mut aabb = self.aggregate.bounding_box();
        if let Some(transform) = self.transform {
            aabb = transform * aabb
        }
        aabb
    }
}

impl Hittable for Instance {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        if let Some(transform) = self.transform {
            if let Some(hit) = self.aggregate.hit(transform.inverse() * r, t0, t1) {
                Some(HitRecord {
                    normal: transform * hit.normal,
                    point: transform * hit.point,
                    instance_id: self.instance_id,
                    ..hit
                })
            } else {
                None
            }
        } else {
            self.aggregate.hit(r, t0, t1)
        }
    }
    fn sample(&self, s: &mut Box<dyn Sampler>, from: Point3) -> (Vec3, PDF) {
        if let Some(transform) = self.transform {
            let (vec, pdf) = self.aggregate.sample(s, transform.reverse * from);
            (transform * vec, pdf)
        } else {
            self.aggregate.sample(s, from)
        }
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        if let Some(transform) = self.transform {
            let (point, normal, pdf) = self.aggregate.sample_surface(s);
            (transform * point, transform * normal, pdf)
        } else {
            self.aggregate.sample_surface(s)
        }
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        let (normal, from, to) = if let Some(transform) = self.transform {
            (
                transform.reverse * normal,
                transform.reverse * from,
                transform.reverse * to,
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
    fn get_instance_id(&self) -> usize {
        self.instance_id
    }
    fn get_material_id(&self) -> MaterialId {
        self.material_id
    }
}

impl From<Aggregate> for Instance {
    fn from(data: Aggregate) -> Self {
        // a direct conversion directly copies the instance id and material id.
        // take care when duplicating instances that are referred to by lights.
        let instance_id = (&data).get_instance_id();
        let material_id = (&data).get_material_id();
        Instance::new(data, None, Some(material_id), Some(instance_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_aggregate() {
        let sphere = Sphere::new(1.0, Point3::ORIGIN, 0, 0);
        let aarect = AARect::new((1.0, 1.0), Point3::ORIGIN, Axis::X, true, 0, 0);

        let transform = Transform3::stack(
            Some(Transform3::translation(Vec3::new(1.0, 1.0, 1.0))),
            Some(Transform3::axis_angle(Vec3::Z, 1.0)),
            Some(Transform3::scale(Vec3::new(3.0, 3.0, 3.0))),
        );

        let aggregate1 = Aggregate::from(sphere);
        let aggregate2 = Aggregate::from(aarect);

        let instance1 = Instance::new(aggregate1, Some(transform), None, None);
        let instance2 = Instance::new(aggregate2, Some(transform), None, None);

        let test_ray = Ray::new(Point3::ORIGIN + 10.0 * Vec3::Z, -Vec3::Z);

        let isect1 = instance1.hit(test_ray, 0.0, 1.0);
        let isect2 = instance2.hit(test_ray, 0.0, 1.0);

        if let Some(hit) = isect1 {
            println!("{:?}", hit);
        }

        if let Some(hit) = isect2 {
            println!("{:?}", hit);
        }
    }
}
