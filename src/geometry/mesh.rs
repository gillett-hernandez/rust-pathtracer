use crate::aabb::AABB;
use crate::geometry::triangle::Triangle;
use crate::hittable::{HasBoundingBox, HitRecord, Hittable};
use crate::materials::MaterialId;
use crate::math::*;

pub struct Mesh {
    // pub triangles: Vec<Triangle>,
    pub num_faces: usize,
    pub v_indices: Vec<usize>,
    pub vertices: Vec<Point3>,
    // pub n_indices: Vec<usize>,
    // pub normals: Vec<Vec3>,
    pub material_ids: Vec<MaterialId>,
    bounding_box: AABB,
}

impl Mesh {
    pub fn new(
        num_faces: usize,
        v_indices: Vec<usize>,
        vertices: Vec<Point3>,
        material_ids: Vec<MaterialId>,
    ) -> Self {
        let mut bounding_box = AABB::empty();
        for tri in 0..num_faces {
            bounding_box.expand_mut(&Triangle::new(tri, &vertices, &v_indices).aabb());
        }

        Mesh {
            num_faces,
            v_indices,
            vertices,
            material_ids,
            bounding_box,
        }
    }
    pub fn get_triangle(&self, tri_num: usize) -> Triangle {
        Triangle::new(tri_num, &self.vertices, &self.v_indices)
    }
}

impl HasBoundingBox for Mesh {
    fn aabb(&self) -> AABB {
        self.bounding_box
    }
}
