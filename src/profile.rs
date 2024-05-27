#[derive(Copy, Clone, Default)]
pub struct Profile {
    pub bounce_rays: usize, // all rays that bounce
    pub shadow_rays: usize, // all rays used to test visibility
    pub light_rays: usize,  // all rays from or to lights
    pub camera_rays: usize, // all rays from or to the camera
    pub env_hits: usize,
}

impl Profile {
    pub fn new(
        bounce_rays: usize,
        shadow_rays: usize,
        light_rays: usize,
        camera_rays: usize,
        env_hits: usize,
    ) -> Self {
        Profile {
            bounce_rays,
            shadow_rays,
            light_rays,
            camera_rays,
            env_hits,
        }
    }
    pub fn combine(&self, other: Self) -> Self {
        Profile::new(
            self.bounce_rays + other.bounce_rays,
            self.shadow_rays + other.shadow_rays,
            self.light_rays + other.light_rays,
            self.camera_rays + other.camera_rays,
            self.env_hits + other.env_hits,
        )
    }

    pub fn pretty_print(&self, elapsed: f32, threads: usize) {
        let &Profile {
            bounce_rays,
            shadow_rays,
            light_rays,
            camera_rays,
            env_hits,
        } = self;
        let sum = bounce_rays + shadow_rays + light_rays + camera_rays;
        println!(
            "{} total bounce rays at {} per second and {} per second per thread",
            bounce_rays,
            bounce_rays as f32 / elapsed,
            bounce_rays as f32 / elapsed / (threads as f32)
        );
        println!(
            "{} total shadow/visibility rays at {} per second and {} per second per thread",
            shadow_rays,
            shadow_rays as f32 / elapsed,
            shadow_rays as f32 / elapsed / (threads as f32)
        );
        println!(
            "{} total light rays at {} per second and {} per second per thread",
            light_rays,
            light_rays as f32 / elapsed,
            light_rays as f32 / elapsed / (threads as f32)
        );
        println!(
            "{} total camera rays at {} per second and {} per second per thread",
            camera_rays,
            camera_rays as f32 / elapsed,
            camera_rays as f32 / elapsed / (threads as f32)
        );
        println!(
            "{} total env hits, {} per second",
            env_hits,
            env_hits as f32 / elapsed,
        );
        println!(
            "{} total rays at {} per second and {} per second per thread",
            sum,
            sum as f32 / elapsed,
            sum as f32 / elapsed / (threads as f32)
        );
    }
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_default_macro() {
        let empty = Profile::default();
    }
}
