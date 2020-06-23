use crate::aabb::*;
use crate::hittable::{HitRecord, Hittable, Indexable};
use crate::math::*;

/*

bool hittable_list::bounding_box(float t0, float t1, aabb &box) const
{
    if (list_size < 1)
    {
        return false;
    }
    aabb temp_box;
    bool first_true = list[0]->bounding_box(t0, t1, temp_box);
    if (!first_true)
    {
        return false;
    }
    else
    {
        box = temp_box;
    }
    for (int i = 1; i < list_size; i++)
    {
        if (list[i]->bounding_box(t0, t1, temp_box))
        {
            box = surrounding_box(box, temp_box);
        }
        else
        {
            return false;
        }
    }
    return true;
}*/
pub struct HittableList {
    pub list: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    pub fn new(list: Vec<Box<dyn Hittable>>) -> HittableList {
        HittableList { list }
    }
}

impl HasBoundingBox for HittableList {
    fn bounding_box(&self) -> AABB {
        let mut bounding_box: Option<AABB> = None;
        for hittable in &self.list {
            if (&bounding_box).is_none() {
                bounding_box = Some(hittable.bounding_box());
            } else {
                bounding_box = Some(bounding_box.unwrap().expand(hittable.bounding_box()));
            }
        }
        bounding_box.unwrap()
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: Ray, t0: f32, t1: f32) -> Option<HitRecord> {
        // let mut hit_anything = false;
        let mut closest_so_far: f32 = t1;
        let mut hit_record: Option<HitRecord> = None;
        for hittable in &self.list {
            // if !hittable.bounding_box().hit(r, t0, closest_so_far) {
            //     continue;
            // }
            let tmp_hit_record = hittable.hit(r, t0, closest_so_far);
            if let Some(hit) = &tmp_hit_record {
                closest_so_far = hit.time;
                hit_record = tmp_hit_record;
            } else {
                continue;
            }
        }
        hit_record
    }
    fn sample(&self, _s: &mut Box<dyn Sampler>, _from: Point3) -> (Vec3, f32) {
        unimplemented!()
    }
    fn pdf(&self, _normal: Vec3, _from: Point3, _to: Point3) -> f32 {
        unimplemented!()
    }
    fn get_instance_id(&self) -> usize {
        unimplemented!()
    }
}

impl Indexable for HittableList {
    fn get_primitive(&self, index: usize) -> &Box<dyn Hittable> {
        &self.list[index]
    }
}
