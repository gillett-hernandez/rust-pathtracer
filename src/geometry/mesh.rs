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
        0 => shuffle!(vec, [1, 2, 0, 3]),
        1 => shuffle!(vec, [2, 0, 1, 3]),
        2 => shuffle!(vec, [0, 1, 2, 3]),
        _ => vec,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshTriangleRef {
    pub idx: usize,
    pub node_index: usize,
    vertices: Arc<Vec<Point3>>,
    indices: Arc<Vec<usize>>,
    normals: Arc<Vec<Vec3>>,
    materials: Arc<Vec<MaterialId>>,
}

impl MeshTriangleRef {
    pub fn new(
        idx: usize,
        vertices: Arc<Vec<Point3>>,
        indices: Arc<Vec<usize>>,
        normals: Arc<Vec<Vec3>>,
        materials: Arc<Vec<MaterialId>>,
    ) -> MeshTriangleRef {
        MeshTriangleRef {
            idx,
            node_index: 0,
            vertices,
            indices,
            normals,
            materials,
        }
    }
    pub fn get_material_id(&self) -> MaterialId {
        if self.materials.len() > 0 {
            self.materials[self.idx]
        } else {
            0u16.into()
        }
    }
}

impl HasBoundingBox for MeshTriangleRef {
    fn aabb(&self) -> AABB {
        let p0 = self.vertices[self.indices[3 * self.idx]];
        let p1 = self.vertices[self.indices[3 * self.idx + 1]];
        let p2 = self.vertices[self.indices[3 * self.idx + 2]];
        AABB::new(p0, p1).grow(&p2)
    }
}

impl Hittable for MeshTriangleRef {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        let p0 = self.vertices[self.indices[3 * self.idx]];
        let p1 = self.vertices[self.indices[3 * self.idx + 1]];
        let p2 = self.vertices[self.indices[3 * self.idx + 2]];
        let mat_id = if self.materials.len() > 0 {
            self.materials[self.idx]
        } else {
            0u16.into()
        };
        let mut p0t = p0 - r.origin;
        let mut p1t = p1 - r.origin;
        let mut p2t = p2 - r.origin;
        let direction = r.direction.0;
        let max_axis_value = direction.abs().max_element();
        let mask = direction.abs().ge(f32x4::splat(max_axis_value));
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
        debug_assert!(
            sx.is_finite() && sy.is_finite() && sz.is_finite(),
            "{:?} {:?} {:?}",
            dx,
            dy,
            dz
        );
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
        debug_assert!(
            !t_scaled.is_nan(),
            "{:?} {:?} {:?} {:?} {:?} {:?}",
            e0,
            p0t_z,
            e1,
            p1t_z,
            e2,
            p2t_z
        );

        if (det < 0.0 && (t_scaled >= t0 * det || t_scaled < t1 * det))
            || (det > 0.0 && (t_scaled <= t0 * det || t_scaled > t1 * det))
        {
            return None;
        }

        let inv_det = det.recip();
        debug_assert!(!inv_det.is_nan(), "{:?}", det);
        let b0 = e0 * inv_det;
        let b1 = e1 * inv_det;
        let b2 = e2 * inv_det;

        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let geometric_normal = dp02.cross(dp12).normalized();
        // shading normal effectively causes smooth shading too, even if the shading normals match the geometric normals.

        let shading_normal = if self.normals.len() > 0 {
            let (n0, n1, n2) = (
                self.normals[self.indices[3 * self.idx]],
                self.normals[self.indices[3 * self.idx + 1]],
                self.normals[self.indices[3 * self.idx + 2]],
            );
            // compute shading tangent and bitangent here too.
            Some(b0 * n0 + b1 * n1 + b2 * n2)
        } else {
            None
        };
        let hit = HitRecord::new(
            t_scaled * inv_det,
            Point3::from(b0 * Vec3::from(p0) + b1 * Vec3::from(p1) + b2 * Vec3::from(p2)),
            (0.0, 0.0),
            0.0,
            shading_normal.unwrap_or(geometric_normal),
            mat_id,
            0,
            None,
        );
        debug_assert!(
            hit.point.is_finite() && hit.normal.is_finite() && hit.time.is_finite(),
            "{:?}, {}, {}",
            hit,
            t_scaled,
            inv_det
        );
        Some(hit)
    }

    fn sample(&self, _s: Sample2D, _from: Point3) -> (Vec3, PDF) {
        unimplemented!("mesh light sampling methods are currently unimplemented")
    }

    fn sample_surface(&self, _s: Sample2D) -> (Point3, Vec3, PDF) {
        unimplemented!("mesh light sampling methods are currently unimplemented")
    }

    fn psa_pdf(&self, _cos_o: f32, _from: Point3, _to: Point3) -> PDF {
        unimplemented!("mesh light sampling methods are currently unimplemented")
    }
    fn surface_area(&self, transform: &Transform3) -> f32 {
        // calculates the surface area using heron's formula.
        let p0 = transform.to_world(self.vertices[self.indices[3 * self.idx]]);
        let p1 = transform.to_world(self.vertices[self.indices[3 * self.idx + 1]]);
        let p2 = transform.to_world(self.vertices[self.indices[3 * self.idx + 2]]);
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
    pub indices: Arc<Vec<usize>>,
    pub vertices: Arc<Vec<Point3>>,
    pub normals: Arc<Vec<Vec3>>,
    pub material_ids: Arc<Vec<MaterialId>>,
    bounding_box: AABB,
    pub bvh: Option<FlatBVH>,
    pub triangles: Option<Vec<MeshTriangleRef>>,
}

impl Mesh {
    pub fn new(
        num_faces: usize,
        v_indices: Vec<usize>,
        vertices: Vec<Point3>,
        normals: Vec<Vec3>,
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
            indices: Arc::new(v_indices),
            vertices: Arc::new(vertices),
            normals: Arc::new(normals),
            material_ids: Arc::new(material_ids),
            bounding_box,
            bvh: None,
            triangles: None,
        }
    }
    pub fn init(&mut self) {
        if self.triangles.is_some() {
            // already initialized
            return;
        }
        let mut triangles = Vec::new();
        for tri_num in 0..self.num_faces {
            triangles.push(MeshTriangleRef::new(
                tri_num,
                Arc::clone(&self.vertices),
                Arc::clone(&self.indices),
                Arc::clone(&self.normals),
                Arc::clone(&self.material_ids),
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
        let bvh = self.bvh.as_ref().expect("bvh not initialized for mesh");
        let mut possible_hit_triangles = bvh.traverse(&r, self.triangles.as_ref().unwrap());
        // maybe sort AABB intersections so that the earliest aabb hit end is first.
        // TODO: run a performance test to see if doing this speeds up renders with meshs
        if cfg!(feature = "sort_mesh_aabb_hits") {
            possible_hit_triangles.sort_unstable_by(
                |(_, _, aabb_hit_end_time0), (_, _, aabb_hit_end_time1)| {
                    // let hit0_t1 = a.2;
                    // let hit1_t1 = b.2;
                    // let sign = (hit1_t1-hit0_t1).signum();
                    (aabb_hit_end_time0)
                        .partial_cmp(aabb_hit_end_time1)
                        .unwrap()
                },
            );
        }
        let mut closest_so_far: f32 = t1;
        let mut hit_record: Option<HitRecord> = None;
        for (tri, t0_aabb_hit, t1_aabb_hit) in possible_hit_triangles {
            if t1_aabb_hit < t0 || t0_aabb_hit > t1 {
                // if bounding box hit was outside of hit time bounds
                continue;
            }
            if t0_aabb_hit > closest_so_far {
                // ignore aabb hit that happened after closest so far
                continue;
            }
            // let t0 = t0.max(t0_aabb_hit);

            // let t1 = closest_so_far.min(t1_aabb_hit);
            // let tmp_hit_record = tri.hit(r, t0, t1);
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
    // TODO: implement mesh and triangle light sampling
    fn sample(&self, _s: Sample2D, _from: Point3) -> (Vec3, PDF) {
        unimplemented!("mesh light sampling is unimplemented")
    }
    fn sample_surface(&self, _s: Sample2D) -> (Point3, Vec3, PDF) {
        unimplemented!("mesh light sampling is unimplemented")
    }
    fn psa_pdf(&self, _cos_o: f32, _from: Point3, _to: Point3) -> PDF {
        unimplemented!("mesh light sampling is unimplemented")
    }
    fn surface_area(&self, _transform: &Transform3) -> f32 {
        unimplemented!("mesh light sampling is unimplemented")
    }
}
/*
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsing::primitives::parse_mesh;
    #[test]
    fn test_mesh() {
        let mut mesh = parse_mesh("data/meshes/monkey.obj", 0);
        mesh.init();
    }
} */
