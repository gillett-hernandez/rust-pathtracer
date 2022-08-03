use crate::aabb::{HasBoundingBox, AABB};
// use crate::bounding_hierarchy::{BHShape, BoundingHierarchy};
use crate::math::Ray;

use std::f32;

use packed_simd::{f32x4, i32x4};

pub trait BHShape: HasBoundingBox {
    /// Sets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
    ///
    fn set_bh_node_index(&mut self, _: usize);

    /// Gets the index of the referenced [`BoundingHierarchy`] node.
    ///
    /// [`BoundingHierarchy`]: struct.BoundingHierarchy.html
    ///
    fn bh_node_index(&self) -> usize;
}

pub trait BoundingHierarchy {
    fn build<Shape: BHShape>(shapes: &mut [Shape]) -> Self;

    fn traverse<'a, Shape: BHShape>(
        &'a self,
        ray: &Ray,
        shapes: &'a [Shape],
    ) -> Vec<(&Shape, f32, f32)>;

    fn pretty_print(&self) {}
}

pub fn concatenate_vectors<T: Sized>(vectors: &mut [Vec<T>]) -> Vec<T> {
    let mut result = Vec::new();
    for vector in vectors.iter_mut() {
        result.append(vector);
    }
    result
}

#[derive(Copy, Clone)]
pub struct Bucket {
    /// The number of shapes in this `Bucket`.
    pub size: usize,

    /// The joint `AABB` of the shapes in this `Bucket`.
    pub aabb: AABB,
}

impl Bucket {
    /// Returns an empty bucket.
    pub fn empty() -> Bucket {
        Bucket {
            size: 0,
            aabb: AABB::empty(),
        }
    }

    /// Extend this `Bucket` by a shape with the given `AABB`.
    pub fn add_aabb(&mut self, aabb: &AABB) {
        self.size += 1;
        self.aabb = self.aabb.expand(aabb);
    }

    /// Join the contents of two `Bucket`s.
    pub fn join_bucket(a: Bucket, b: &Bucket) -> Bucket {
        Bucket {
            size: a.size + b.size,
            aabb: a.aabb.expand(&b.aabb),
        }
    }
}

pub fn joint_aabb_of_shapes<Shape: BHShape>(indices: &[usize], shapes: &[Shape]) -> AABB {
    let mut aabb = AABB::empty();
    for index in indices {
        let shape = &shapes[*index];
        aabb.expand_mut(&shape.aabb());
    }
    aabb
}

/// The [`BVHNode`] enum that describes a node in a [`BVH`].
/// It's either a leaf node and references a shape (by holding its index)
/// or a regular node that has two child nodes.
/// The non-leaf node stores the [`AABB`]s of its children.
///
/// [`AABB`]: ../aabb/struct.AABB.html
/// [`BVH`]: struct.BVH.html
/// [`BVH`]: struct.BVHNode.html
///
#[derive(Debug, Copy, Clone)]
pub enum BVHNode {
    /// Leaf node.
    Leaf {
        /// The node's parent.
        parent_index: usize,

        /// The node's depth.
        depth: u32,

        /// The shape contained in this leaf.
        shape_index: usize,
    },
    /// Inner node.
    Node {
        /// The node's parent.
        parent_index: usize,

        /// The node's depth.
        depth: u32,

        /// Index of the left subtree's root node.
        child_l_index: usize,

        /// The convex hull of the shapes' `AABB`s in child_l.
        child_l_aabb: AABB,

        /// Index of the right subtree's root node.
        child_r_index: usize,

        /// The convex hull of the shapes' `AABB`s in child_r.
        child_r_aabb: AABB,
    },
}

impl PartialEq for BVHNode {
    // TODO Consider also comparing AABBs
    fn eq(&self, other: &BVHNode) -> bool {
        match (self, other) {
            (
                &BVHNode::Node {
                    parent_index: self_parent_index,
                    depth: self_depth,
                    child_l_index: self_child_l_index,
                    child_r_index: self_child_r_index,
                    ..
                },
                &BVHNode::Node {
                    parent_index: other_parent_index,
                    depth: other_depth,
                    child_l_index: other_child_l_index,
                    child_r_index: other_child_r_index,
                    ..
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_depth == other_depth
                    && self_child_l_index == other_child_l_index
                    && self_child_r_index == other_child_r_index
            }
            (
                &BVHNode::Leaf {
                    parent_index: self_parent_index,
                    depth: self_depth,
                    shape_index: self_shape_index,
                },
                &BVHNode::Leaf {
                    parent_index: other_parent_index,
                    depth: other_depth,
                    shape_index: other_shape_index,
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_depth == other_depth
                    && self_shape_index == other_shape_index
            }
            _ => false,
        }
    }
}

impl BVHNode {
    /// Returns the index of the parent node.
    pub fn parent(&self) -> usize {
        match *self {
            BVHNode::Node { parent_index, .. } | BVHNode::Leaf { parent_index, .. } => parent_index,
        }
    }

    /// Returns a mutable reference to the parent node index.
    pub fn parent_mut(&mut self) -> &mut usize {
        match *self {
            BVHNode::Node {
                ref mut parent_index,
                ..
            }
            | BVHNode::Leaf {
                ref mut parent_index,
                ..
            } => parent_index,
        }
    }

    /// Returns the index of the left child node.
    pub fn child_l(&self) -> usize {
        match *self {
            BVHNode::Node { child_l_index, .. } => child_l_index,
            _ => panic!("Tried to get the left child of a leaf node."),
        }
    }

    /// Returns the `AABB` of the right child node.
    pub fn child_l_aabb(&self) -> AABB {
        match *self {
            BVHNode::Node { child_l_aabb, .. } => child_l_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the `AABB` of the left child node.
    pub fn child_l_aabb_mut(&mut self) -> &mut AABB {
        match *self {
            BVHNode::Node {
                ref mut child_l_aabb,
                ..
            } => child_l_aabb,
            _ => panic!("Tried to get the left child's `AABB` of a leaf node."),
        }
    }

    /// Returns the index of the right child node.
    pub fn child_r(&self) -> usize {
        match *self {
            BVHNode::Node { child_r_index, .. } => child_r_index,
            _ => panic!("Tried to get the right child of a leaf node."),
        }
    }

    /// Returns the `AABB` of the right child node.
    pub fn child_r_aabb(&self) -> AABB {
        match *self {
            BVHNode::Node { child_r_aabb, .. } => child_r_aabb,
            _ => panic!(),
        }
    }

    /// Returns a mutable reference to the `AABB` of the right child node.
    pub fn child_r_aabb_mut(&mut self) -> &mut AABB {
        match *self {
            BVHNode::Node {
                ref mut child_r_aabb,
                ..
            } => child_r_aabb,
            _ => panic!("Tried to get the right child's `AABB` of a leaf node."),
        }
    }

    /// Returns the depth of the node. The root node has depth `0`.
    pub fn depth(&self) -> u32 {
        match *self {
            BVHNode::Node { depth, .. } | BVHNode::Leaf { depth, .. } => depth,
        }
    }

    /// Gets the `AABB` for a `BVHNode`.
    /// Returns the shape's `AABB` for leaves, and the joined `AABB` of
    /// the two children's `AABB`s for non-leaves.
    pub fn get_node_aabb<Shape: BHShape>(&self, shapes: &[Shape]) -> AABB {
        match *self {
            BVHNode::Node {
                child_l_aabb,
                child_r_aabb,
                ..
            } => child_l_aabb.expand(&child_r_aabb),
            BVHNode::Leaf { shape_index, .. } => shapes[shape_index].aabb(),
        }
    }

    /// Returns the index of the shape contained within the node if is a leaf,
    /// or `None` if it is an interior node.
    pub fn shape_index(&self) -> Option<usize> {
        match *self {
            BVHNode::Leaf { shape_index, .. } => Some(shape_index),
            _ => None,
        }
    }

    /// The build function sometimes needs to add nodes while their data is not available yet.
    /// A dummy cerated by this function serves the purpose of being changed later on.
    fn create_dummy() -> BVHNode {
        BVHNode::Leaf {
            parent_index: 0,
            depth: 0,
            shape_index: 0,
        }
    }

    /// Builds a [`BVHNode`] recursively using SAH partitioning.
    /// Returns the index of the new node in the nodes vector.
    ///
    /// [`BVHNode`]: enum.BVHNode.html
    ///
    pub fn build<T: BHShape>(
        shapes: &mut [T],
        indices: &[usize],
        nodes: &mut Vec<BVHNode>,
        parent_index: usize,
        depth: u32,
    ) -> usize {
        // Helper function to accumulate the AABB joint and the centroids AABB
        fn grow_convex_hull(convex_hull: (AABB, AABB), shape_aabb: &AABB) -> (AABB, AABB) {
            let center = &shape_aabb.center();
            let convex_hull_aabbs = &convex_hull.0;
            let convex_hull_centroids = &convex_hull.1;
            let expanded_box = convex_hull_aabbs.expand(shape_aabb);
            let expanded_centroids = convex_hull_centroids.grow(center);
            // println!(
            //     "shape center and box was {:?} and {:?}, original box was {:?}, expanded aabb was {:?}, original centroid was {:?}, expanded centroid was {:?}",
            //     center, shape_aabb, convex_hull_aabbs, expanded_box, convex_hull_centroids, expanded_centroids
            // );
            (expanded_box, expanded_centroids)
        }

        let mut convex_hull = Default::default();
        for index in indices {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = convex_hull;

        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            let node_index = nodes.len();
            nodes.push(BVHNode::Leaf {
                parent_index,
                depth,
                shape_index,
            });
            // Let the shape know the index of the node that represents it.
            shapes[shape_index].set_bh_node_index(node_index);
            return node_index;
        }

        // From here on we handle the recursive case. This dummy is required, because the children
        // must know their parent, and it's easier to update one parent node than the child nodes.
        let node_index = nodes.len();
        nodes.push(BVHNode::create_dummy());

        // Find the axis along which the shapes are spread the most.
        let size = centroid_bounds.size().0;
        let max_axis = size.max_element();
        let mask = size.ge(f32x4::splat(max_axis));
        let split_axis = mask
            .select(i32x4::new(0, 1, 2, 3), i32x4::splat(0))
            .max_element() as usize;
        // println!(
        //     "size: {:?}, max_axis: {:?}, split_axis: {:?}",
        //     size, max_axis, split_axis
        // );
        // let split_axis = split_axis;
        let split_axis_size =
            centroid_bounds.max.0.extract(split_axis) - centroid_bounds.min.0.extract(split_axis);

        // The following `if` partitions `indices` for recursively calling `BVH::build`.
        let (child_l_index, child_l_aabb, child_r_index, child_r_aabb) = if split_axis_size
            < 0.00001
        {
            // In this branch the shapes lie too close together so that splitting them in a
            // sensible way is not possible. Instead we just split the list of shapes in half.
            let (child_l_indices, child_r_indices) = indices.split_at(indices.len() / 2);

            // println!("h {}, {}", depth, indices.len(),);

            let child_l_aabb = joint_aabb_of_shapes(child_l_indices, shapes);
            let child_r_aabb = joint_aabb_of_shapes(child_r_indices, shapes);

            // Proceed recursively.
            let child_l_index =
                BVHNode::build(shapes, child_l_indices, nodes, node_index, depth + 1);
            let child_r_index =
                BVHNode::build(shapes, child_r_indices, nodes, node_index, depth + 1);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        } else {
            // Create six `Bucket`s, and six index assignment vector.
            const NUM_BUCKETS: usize = 6;
            let mut buckets = [Bucket::empty(); NUM_BUCKETS];
            let mut bucket_assignments: [Vec<usize>; NUM_BUCKETS] = Default::default();

            // In this branch the `split_axis_size` is large enough to perform meaningful splits.
            // We start by assigning the shapes to `Bucket`s.
            for idx in indices {
                let shape = &shapes[*idx];
                let shape_aabb = shape.aabb();
                let shape_center = shape_aabb.center();

                // Get the relative position of the shape centroid `[0.0..1.0]`.
                let bucket_num_relative = (shape_center.0.extract(split_axis)
                    - centroid_bounds.min.0.extract(split_axis))
                    / split_axis_size;

                // Convert that to the actual `Bucket` number.
                let bucket_num = (bucket_num_relative * (NUM_BUCKETS as f32 - 0.01)) as usize;

                // Extend the selected `Bucket` and add the index to the actual bucket.
                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }

            // Compute the costs for each configuration and select the best configuration.
            let mut min_bucket = 0;
            let mut min_cost = f32::INFINITY;
            let mut child_l_aabb = AABB::empty();
            let mut child_r_aabb = AABB::empty();
            for i in 0..(NUM_BUCKETS - 1) {
                let (l_buckets, r_buckets) = buckets.split_at(i + 1);
                let child_l = l_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);
                let child_r = r_buckets.iter().fold(Bucket::empty(), Bucket::join_bucket);

                let cost = (child_l.size as f32 * child_l.aabb.surface_area()
                    + child_r.size as f32 * child_r.aabb.surface_area())
                    / aabb_bounds.surface_area();
                if cost < min_cost {
                    min_bucket = i;
                    min_cost = cost;
                    child_l_aabb = child_l.aabb;
                    child_r_aabb = child_r.aabb;
                }
            }

            // Join together all index buckets.
            let (l_assignments, r_assignments) = bucket_assignments.split_at_mut(min_bucket + 1);
            let child_l_indices = concatenate_vectors(l_assignments);
            let child_r_indices = concatenate_vectors(r_assignments);

            // Proceed recursively.
            // println!(
            //     "buckets {}, {} {}",
            //     depth,
            //     child_l_indices.len(),
            //     child_r_indices.len()
            // );
            let child_l_index =
                BVHNode::build(shapes, &child_l_indices, nodes, node_index, depth + 1);
            let child_r_index =
                BVHNode::build(shapes, &child_r_indices, nodes, node_index, depth + 1);
            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        };

        // Construct the actual data structure and replace the dummy node.
        assert!(!child_l_aabb.is_empty());
        assert!(!child_r_aabb.is_empty());
        nodes[node_index] = BVHNode::Node {
            parent_index,
            depth,
            child_l_aabb,
            child_l_index,
            child_r_aabb,
            child_r_index,
        };

        node_index
    }

    /// Traverses the [`BVH`] recursively and returns all shapes whose [`AABB`] is
    /// intersected by the given [`Ray`].
    ///
    /// [`AABB`]: ../aabb/struct.AABB.html
    /// [`BVH`]: struct.BVH.html
    /// [`Ray`]: ../ray/struct.Ray.html
    ///
    pub fn traverse_recursive(
        nodes: &[BVHNode],
        node_index: usize,
        ray: &Ray,
        t0: f32,
        t1: f32,
        indices: &mut Vec<(usize, f32, f32)>,
    ) {
        match nodes[node_index] {
            BVHNode::Node {
                ref child_l_aabb,
                child_l_index,
                ref child_r_aabb,
                child_r_index,
                ..
            } => {
                if let Some((t0, t1)) = child_l_aabb.hit(ray, t0, t1) {
                    BVHNode::traverse_recursive(nodes, child_l_index, ray, t0, t1, indices);
                }
                if let Some((t0, t1)) = child_r_aabb.hit(ray, t0, t1) {
                    BVHNode::traverse_recursive(nodes, child_r_index, ray, t0, t1, indices);
                }
            }
            BVHNode::Leaf { shape_index, .. } => {
                indices.push((shape_index, t0, t1));
            }
        }
    }
}

/// The [`BVH`] data structure. Contains the list of [`BVHNode`]s.
///
/// [`BVH`]: struct.BVH.html
///
#[derive(Clone, Debug)]
pub struct BVH {
    /// The list of nodes of the [`BVH`].
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub nodes: Vec<BVHNode>,
}

impl BVH {
    /// Creates a new [`BVH`] from the `shapes` slice.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn build<Shape: BHShape>(shapes: &mut [Shape]) -> BVH {
        info!("building bvh");
        let indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2;
        let mut nodes = Vec::with_capacity(expected_node_count);
        BVHNode::build(shapes, &indices, &mut nodes, 0, 0);
        BVH { nodes }
    }

    /// Traverses the [`BVH`].
    /// Returns a subset of `shapes`, in which the [`AABB`]s of the elements were hit by `ray`.
    ///
    /// [`BVH`]: struct.BVH.html
    /// [`AABB`]: ../aabb/struct.AABB.html
    ///
    pub fn traverse<'a, Shape: HasBoundingBox>(
        &'a self,
        ray: &Ray,
        shapes: &'a [Shape],
    ) -> Vec<(&Shape, f32, f32)> {
        let mut indices = Vec::new();
        BVHNode::traverse_recursive(&self.nodes, 0, ray, 0.0, f32::INFINITY, &mut indices);
        indices
            .iter()
            .map(|(index, t0, t1)| (&shapes[*index], *t0, *t1))
            .collect::<Vec<_>>()
    }

    /// Prints the [`BVH`] in a tree-like visualization.
    ///
    /// [`BVH`]: struct.BVH.html
    ///
    pub fn pretty_print(&self) {
        let nodes = &self.nodes;
        fn print_node(nodes: &[BVHNode], node_index: usize) {
            match nodes[node_index] {
                BVHNode::Node {
                    child_l_index,
                    child_r_index,
                    depth,
                    child_l_aabb,
                    child_r_aabb,
                    ..
                } => {
                    let padding: String = " ".repeat(depth as usize);
                    println!("{}child_l {:?}", padding, child_l_aabb);
                    print_node(nodes, child_l_index);
                    println!("{}child_r {:?}", padding, child_r_aabb);
                    print_node(nodes, child_r_index);
                }
                BVHNode::Leaf {
                    shape_index, depth, ..
                } => {
                    let padding: String = " ".repeat(depth as usize);
                    println!("{}shape\t{:?}", padding, shape_index);
                }
            }
        }
        print_node(nodes, 0);
    }

    // fn is_consistent_subtree<Shape: BHShape>(
    //     &self,
    //     node_index: usize,
    //     expected_parent_index: usize,
    //     expected_outer_aabb: &AABB,
    //     expected_depth: u32,
    //     node_count: &mut usize,
    //     shapes: &[Shape],
    // ) -> bool {
    //     *node_count += 1;
    //     match self.nodes[node_index] {
    //         BVHNode::Node {
    //             parent_index,
    //             depth,
    //             child_l_index,
    //             child_l_aabb,
    //             child_r_index,
    //             child_r_aabb,
    //         } => {
    //             let correct_parent_index = expected_parent_index == parent_index;
    //             let correct_depth = expected_depth == depth;
    //             let left_aabb_in_parent =
    //                 expected_outer_aabb.approx_contains_aabb_eps(&child_l_aabb, 0.00001);
    //             let right_aabb_in_parent =
    //                 expected_outer_aabb.approx_contains_aabb_eps(&child_r_aabb, 0.00001);
    //             let left_subtree_consistent = self.is_consistent_subtree(
    //                 child_l_index,
    //                 node_index,
    //                 &child_l_aabb,
    //                 expected_depth + 1,
    //                 node_count,
    //                 shapes,
    //             );
    //             let right_subtree_consistent = self.is_consistent_subtree(
    //                 child_r_index,
    //                 node_index,
    //                 &child_r_aabb,
    //                 expected_depth + 1,
    //                 node_count,
    //                 shapes,
    //             );

    //             correct_parent_index
    //                 && correct_depth
    //                 && left_aabb_in_parent
    //                 && right_aabb_in_parent
    //                 && left_subtree_consistent
    //                 && right_subtree_consistent
    //         }
    //         BVHNode::Leaf {
    //             parent_index,
    //             depth,
    //             shape_index,
    //         } => {
    //             let correct_parent_index = expected_parent_index == parent_index;
    //             let correct_depth = expected_depth == depth;
    //             let shape_aabb = shapes[shape_index].aabb();
    //             let shape_aabb_in_parent =
    //                 expected_outer_aabb.approx_contains_aabb_eps(&shape_aabb, 0.00001);

    //             correct_parent_index && correct_depth && shape_aabb_in_parent
    //         }
    //     }
    // }

    // /// Checks if all children of a node have the correct parent index, and that there is no
    // /// detached subtree. Also checks if the `AABB` hierarchy is consistent.
    // pub fn is_consistent<Shape: BHShape>(&self, shapes: &[Shape]) -> bool {
    //     // The root node of the bvh is not HasBoundingBox by anything.
    //     let space = AABB {
    //         min: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
    //         max: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
    //     };

    //     // The counter for all nodes.
    //     let mut node_count = 0;
    //     let subtree_consistent =
    //         self.is_consistent_subtree(0, 0, &space, 0, &mut node_count, shapes);

    //     // Check if all nodes have been counted from the root node.
    //     // If this is false, it means we have a detached subtree.
    //     let is_connected = node_count == self.nodes.len();
    //     subtree_consistent && is_connected
    // }

    // /// Assert version of `is_consistent_subtree`.
    // fn assert_consistent_subtree<Shape: BHShape>(
    //     &self,
    //     node_index: usize,
    //     expected_parent_index: usize,
    //     expected_outer_aabb: &AABB,
    //     expected_depth: u32,
    //     node_count: &mut usize,
    //     shapes: &[Shape],
    // ) {
    //     *node_count += 1;
    //     let node = &self.nodes[node_index];

    //     let parent = node.parent();
    //     assert_eq!(
    //         expected_parent_index, parent,
    //         "Wrong parent index. Expected: {}; Actual: {}",
    //         expected_parent_index, parent
    //     );
    //     let depth = node.depth();
    //     assert_eq!(
    //         expected_depth, depth,
    //         "Wrong depth. Expected: {}; Actual: {}",
    //         expected_depth, depth
    //     );

    //     match *node {
    //         BVHNode::Node {
    //             child_l_index,
    //             child_l_aabb,
    //             child_r_index,
    //             child_r_aabb,
    //             ..
    //         } => {
    //             assert!(
    //                 expected_outer_aabb.approx_contains_aabb_eps(&child_l_aabb, 0.00001),
    //                 "Left child lies outside the expected bounds.
    //                      \tBounds: {}
    //                      \tLeft child: {}",
    //                 expected_outer_aabb,
    //                 child_l_aabb
    //             );
    //             assert!(
    //                 expected_outer_aabb.approx_contains_aabb_eps(&child_r_aabb, 0.00001),
    //                 "Right child lies outside the expected bounds.
    //                      \tBounds: {}
    //                      \tRight child: {}",
    //                 expected_outer_aabb,
    //                 child_r_aabb
    //             );
    //             self.assert_consistent_subtree(
    //                 child_l_index,
    //                 node_index,
    //                 &child_l_aabb,
    //                 expected_depth + 1,
    //                 node_count,
    //                 shapes,
    //             );
    //             self.assert_consistent_subtree(
    //                 child_r_index,
    //                 node_index,
    //                 &child_r_aabb,
    //                 expected_depth + 1,
    //                 node_count,
    //                 shapes,
    //             );
    //         }
    //         BVHNode::Leaf { shape_index, .. } => {
    //             let shape_aabb = shapes[shape_index].aabb();
    //             assert!(
    //                 expected_outer_aabb.approx_contains_aabb_eps(&shape_aabb, 0.00001),
    //                 "Shape's AABB lies outside the expected bounds.\n\tBounds: {}\n\tShape: {}",
    //                 expected_outer_aabb,
    //                 shape_aabb
    //             );
    //         }
    //     }
    // }

    // /// Assert version of `is_consistent`.
    // pub fn assert_consistent<Shape: BHShape>(&self, shapes: &[Shape]) {
    //     // The root node of the bvh is not HasBoundingBox by anything.
    //     let space = AABB {
    //         min: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
    //         max: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
    //     };

    //     // The counter for all nodes.
    //     let mut node_count = 0;
    //     self.assert_consistent_subtree(0, 0, &space, 0, &mut node_count, shapes);

    //     // Check if all nodes have been counted from the root node.
    //     // If this is false, it means we have a detached subtree.
    //     assert_eq!(node_count, self.nodes.len(), "Detached subtree");
    // }

    // pub fn assert_tight_subtree<Shape: BHShape>(
    //     &self,
    //     node_index: usize,
    //     outer_aabb: &AABB,
    //     shapes: &[Shape],
    // ) {
    //     if let BVHNode::Node {
    //         child_l_index,
    //         child_l_aabb,
    //         child_r_index,
    //         child_r_aabb,
    //         ..
    //     } = self.nodes[node_index]
    //     {
    //         let joint_aabb = child_l_aabb.expand(&child_r_aabb);
    //         assert!(joint_aabb.relative_eq(outer_aabb, 0.00001));
    //         self.assert_tight_subtree(child_l_index, &child_l_aabb, shapes);
    //         self.assert_tight_subtree(child_r_index, &child_r_aabb, shapes);
    //     }
    // }

    // /// Check that the `AABB`s in the `BVH` are tight, which means, that parent `AABB`s are not
    // /// larger than they should be.
    // pub fn assert_tight<Shape: BHShape>(&self, shapes: &[Shape]) {
    //     // When starting to check whether the `BVH` is tight, we cannot provide a minimum
    //     // outer `AABB`, therefore we compute the correct one in this instance.
    //     if let BVHNode::Node {
    //         child_l_aabb,
    //         child_r_aabb,
    //         ..
    //     } = self.nodes[0]
    //     {
    //         let joint_aabb = child_l_aabb.expand(&child_r_aabb);
    //         self.assert_tight_subtree(0, &joint_aabb, shapes);
    //     }
    // }
}

impl BoundingHierarchy for BVH {
    fn build<Shape: BHShape>(shapes: &mut [Shape]) -> BVH {
        BVH::build(shapes)
    }

    fn traverse<'a, Shape: HasBoundingBox>(
        &'a self,
        ray: &Ray,
        shapes: &'a [Shape],
    ) -> Vec<(&Shape, f32, f32)> {
        self.traverse(ray, shapes)
    }

    fn pretty_print(&self) {
        self.pretty_print();
    }
}
