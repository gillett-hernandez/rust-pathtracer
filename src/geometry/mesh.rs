use crate::aabb::AABB;
use crate::accelerator::{BHShape, BoundingHierarchy, FlatBVH};
// use crate::geometry::triangle::Triangle;
use crate::hittable::{HasBoundingBox, HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

use packed_simd::{f32x4, i32x4};

use std::sync::Arc;

pub fn vec_shuffle(vec: f32x4, m: u32) -> f32x4 {
    match m {
        0 => shuffle!(vec, [0, 1, 2, 3]),
        1 => shuffle!(vec, [1, 0, 2, 3]),
        2 => shuffle!(vec, [2, 0, 1, 3]),
        _ => vec,
    }
}

#[derive(Clone, Debug, PartialEq)]
struct MeshTriangleRef {
    pub idx: usize,
    pub node_index: usize,
    vertices: Arc<Vec<Point3>>,
    v_indices: Arc<Vec<usize>>,
}

impl MeshTriangleRef {
    pub fn new(
        idx: usize,
        vertices: Arc<Vec<Point3>>,
        v_indices: Arc<Vec<usize>>,
    ) -> MeshTriangleRef {
        MeshTriangleRef {
            idx,
            node_index: 0,
            vertices,
            v_indices,
        }
    }
}

impl HasBoundingBox for MeshTriangleRef {
    fn aabb(&self) -> AABB {
        let p0 = self.vertices[self.v_indices[3 * self.idx + 0]];
        let p1 = self.vertices[self.v_indices[3 * self.idx + 1]];
        let p2 = self.vertices[self.v_indices[3 * self.idx + 2]];
        AABB::new(p0, p1).grow(&p2)
    }
}

impl Hittable for MeshTriangleRef {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let p0 = self.vertices[self.v_indices[3 * self.idx + 0]];
        let p1 = self.vertices[self.v_indices[3 * self.idx + 1]];
        let p2 = self.vertices[self.v_indices[3 * self.idx + 2]];
        let mut p0t = p0 - r.origin;
        let mut p1t = p1 - r.origin;
        let mut p2t = p2 - r.origin;
        let direction = r.direction.0;
        let max_axis_value = direction.max_element();
        let mask = direction.ge(f32x4::splat(max_axis_value));
        let max_axis = mask
            .select(i32x4::new(0, 1, 2, 3), i32x4::splat(0))
            .max_element() as u32;
        let kz = max_axis;
        let d: f32x4 = vec_shuffle(direction, kz);
        let [dx, dy, dz, _]: [f32; 4] = d.into();
        p0t.0 = vec_shuffle(p0t.0, kz);
        p1t.0 = vec_shuffle(p1t.0, kz);
        p2t.0 = vec_shuffle(p2t.0, kz);
        let [mut p0t_x, mut p0t_y, mut p0t_z, _]: [f32; 4] = p0t.0.into();
        let [mut p1t_x, mut p1t_y, mut p1t_z, _]: [f32; 4] = p1t.0.into();
        let [mut p2t_x, mut p2t_y, mut p2t_z, _]: [f32; 4] = p2t.0.into();
        let sx = -dx / dz;
        let sy = -dy / dz;
        let sz = 1.0 / dz;
        p0t_x += sx * p0t_z;
        p1t_x += sx * p1t_z;
        p2t_x += sx * p2t_z;
        p0t_y += sy * p0t_z;
        p1t_y += sy * p1t_z;
        p2t_y += sy * p2t_z;
        let mut e0 = p1t_x * p2t_y - p1t_y * p2t_x;
        let mut e1 = p2t_x * p0t_y - p2t_y * p0t_x;
        let mut e2 = p0t_x * p1t_y - p0t_y * p1t_x;
        if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
            let p2txp1ty: f64 = (p2t_x as f64) * (p1t_y as f64);
            let p2typ1tx: f64 = (p2t_y as f64) * (p1t_x as f64);
            e0 = (p2typ1tx - p2txp1ty) as f32;

            let p0txp2ty: f64 = (p0t_x as f64) * (p2t_y as f64);
            let p0typ2tx: f64 = (p0t_y as f64) * (p2t_x as f64);
            e1 = (p0typ2tx - p0txp2ty) as f32;

            let p1txp0ty: f64 = (p1t_x as f64) * (p0t_y as f64);
            let p1typ0tx: f64 = (p1t_y as f64) * (p0t_x as f64);
            e2 = (p1typ0tx - p1txp0ty) as f32;
        }

        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            return None;
        }
        let det = e0 + e1 + e2;
        if det == 0.0 {
            return None;
        }

        p0t_z *= sz;
        p1t_z *= sz;
        p2t_z *= sz;

        let t_scaled = e0 * p0t_z + e1 * p1t_z + e2 * p2t_z;

        if (det < 0.0 && (t_scaled >= t0 * det || t_scaled < t1 * det))
            || (det > 0.0 && (t_scaled <= t0 * det || t_scaled > t1 * det))
        {
            return None;
        }

        let inv_det = det.recip();
        let b0 = e0 * inv_det;
        let b1 = e1 * inv_det;
        let b2 = e2 * inv_det;

        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        Some(HitRecord::new(
            t_scaled * inv_det,
            Point3::from(b0 * Vec3::from(p0) + b1 * Vec3::from(p1) + b2 * Vec3::from(p2)),
            (0.0, 0.0),
            0.0,
            dp02.cross(dp12).normalized(),
            0.into(),
            0,
            None,
        ))
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        (Vec3::ZERO, 0.0.into())
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        (Point3::ORIGIN, Vec3::ZERO, 0.0.into())
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        0.0.into()
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        // calculates the surface area using heron's formula.
        let p0 = *transform * self.vertices[self.v_indices[3 * self.idx + 0]];
        let p1 = *transform * self.vertices[self.v_indices[3 * self.idx + 1]];
        let p2 = *transform * self.vertices[self.v_indices[3 * self.idx + 2]];
        let d02 = (p2 - p0).norm();
        let d01 = (p1 - p0).norm();
        let d12 = (p2 - p1).norm();
        let s = 0.5 * (d02 + d01 + d12);
        (s * (s - d01) * (s - d12) * (s - d02)).sqrt()
    }
}

impl BHShape for MeshTriangleRef {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub num_faces: usize,
    pub node_indices: Vec<usize>,
    pub v_indices: Arc<Vec<usize>>,
    pub vertices: Arc<Vec<Point3>>,
    pub material_ids: Vec<MaterialId>,
    bounding_box: AABB,
    pub bvh: Option<FlatBVH>,
    triangles: Option<Vec<MeshTriangleRef>>,
}

impl Mesh {
    pub fn new(
        num_faces: usize,
        v_indices: Vec<usize>,
        vertices: Vec<Point3>,
        material_ids: Vec<MaterialId>,
    ) -> Self {
        let mut bounding_box = AABB::empty();
        for point in vertices.iter() {
            bounding_box.grow_mut(point);
        }

        let node_indices = vec![0usize; num_faces];

        Mesh {
            num_faces,
            node_indices,
            v_indices: Arc::new(v_indices),
            vertices: Arc::new(vertices),
            material_ids,
            bounding_box,
            bvh: None,
            triangles: None,
        }
    }
    pub fn init(&mut self) {
        let mut triangles = Vec::new();
        for tri_num in 0..self.num_faces {
            triangles.push(MeshTriangleRef::new(
                tri_num,
                Arc::clone(&self.vertices),
                Arc::clone(&self.v_indices),
            ));
        }

        self.bvh = Some(FlatBVH::build(triangles.as_mut_slice()));
        for (tri_num, tri) in triangles.iter().enumerate() {
            self.node_indices[tri_num] = tri.node_index;
        }
        self.triangles = Some(triangles);
    }
}

impl HasBoundingBox for Mesh {
    fn aabb(&self) -> AABB {
        self.bounding_box
    }
}

impl Hittable for Mesh {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let bvh = self.bvh.as_ref().unwrap();
        let possible_hit_triangles = bvh.traverse(&r, &self.triangles.as_ref().unwrap());
        let mut closest_so_far: f32 = t1;
        let mut hit_record: Option<HitRecord> = None;
        for tri in possible_hit_triangles {
            let tmp_hit_record = tri.hit(r, t0, closest_so_far);
            if let Some(hit) = &tmp_hit_record {
                closest_so_far = hit.time;
                hit_record = tmp_hit_record;
            } else {
                continue;
            }
        }
        hit_record
    }
    fn sample(&self, s: Sample2D, from: Point3) -> (Vec3, PDF) {
        (Vec3::ZERO, 0.0.into())
    }
    fn sample_surface(&self, s: Sample2D) -> (Point3, Vec3, PDF) {
        (Point3::ORIGIN, Vec3::ZERO, 0.0.into())
    }
    fn pdf(&self, normal: Vec3, from: Point3, to: Point3) -> PDF {
        0.0.into()
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        0.0
    }
}
