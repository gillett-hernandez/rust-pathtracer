# Rust Pathtracer


This is a wavelength aware physically based 3D rendering engine written in Rust. Performance is not the focus, as the focus is mostly in implementing concepts in a concise and readable manner.

The purpose is to help me become more familiar with Rust and with general Light Transport algorithms. However if it helps the reader learn more about these concepts, that would be great.

Most, if not all of the integrators use importance sampling and MIS, and they trace using single wavelength sampling

It supports the following integrators:
* [Path Tracing](src/integrator/pt.rs), described on wikipedia [here](https://en.wikipedia.org/wiki/Path_tracing)

Work is in progress on other branches for the following integrators:
* [Light tracing](src/integrator/lt.rs), also known as particle tracing, where light is emitted from light sources in the scene and traced until it hits the camera
* [Bidirectional Path Tracing](src/integrator/bdpt/mod.rs), described in PBRT [here](http://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing.html)

In addition, much of the code emphasizes matching reality as closely as possible, including the following features:
* Output format:
  * The renderer outputs an .exr file in Linear RGB space, and a .png file in sRGB space.
    * custom exposure values for the sRGB tonemapper are supported, however they are not part of the config file as of yet. the default behavior is to set the brightest pixel on the screen to white.
* Colors and Lights:
  * Colors on the film are represented in [CIE XYZ](https://en.wikipedia.org/wiki/CIE_1931_color_space) color space. this is then color mapped to linear RGB space and sRGB space according to the [wikipedia article](https://en.wikipedia.org/wiki/SRGB)
  * Lights can have physically correct spectral power distribution functions, including blackbody distributions and distributions with peaks at certain frequencies
  * Colors are implemented as Spectral Response Functions, under the hood they are bounded spectral power distributions
  * in general, for lights and for colors, those spectral response functions are implemented as curves, and multiple curve types are supported. see [curves.rs](src/curves.rs) and [math/spectral.rs](src/math/spectral.rs) for more information
* Metals and Dielectrics are wavelength-dependent:
  * Dielectrics use a curve (struct SPD) to represent their varying index of refraction with respect to wavelength.
    * This allows for physically correct Dispersion. The curve used is typically a curve matching the first two terms of [Cauchy's equation](https://en.wikipedia.org/wiki/Cauchy%27s_equation).
  * Metals use multiple curves to represent their varying index of refraction and extinction coefficient with respect to wavelength.
    * This allows for physically correct color and reflectance behavior so that Gold, Copper, and other metals can be represented and traced accurately.


## Installing and running:

Requirements are Rust nightly.

Building and running should be as simple as executing `cargo run` from the shell while in the project root directory.

to change render settings, modify the provided file at data/config.toml

it comes preloaded with many options and most are commented out.


## Experimental implementations:

The Light Tracing Integrator and the Bidirectional Path tracing integrator can represent the camera's lens in the scene, and allow for light to intersect the camera lens. While typically this is done with one camera at a time, I've taken the liberty of trying to implement it so that multiple cameras can be in the scene at once, and if light happens to intersect *any* camera, the contribution will be recorded to that film. This is batched so that all the films that use the Light Tracing integrator will all have their cameras in the scene at once, and the same for Bidirectional Path Tracing.

That should theoretically cause images to converge faster, though the feature is still a WIP and may be changed in the future. Renders using the Path tracing integrator will be unaffected, and their cameras will not physically exist in the World.


## To Do: an incomplete list
- [x] implement basic config file to reduce unnecessary recompilations. Done, at [config.rs](src/config.rs), data files at [data/config.toml](data/config.toml)
- [x] add simple random walk abstraction. Done, at [integrator/bdpt/helpers.rs](src/integrator/bdpt/helpers.rs)
- [x] implement glossy and transmissive bsdfs. Done, implemented GGX, at [materials/ggx.rs](src/materials/ggx.rs)
- [x] add common color spectral reflectance functions. Done, implemented at [curves.rs](src/curves.rs)
- [x] implement correct XYZ to sRGB tonemapping. Done, at [tonemap.rs](src/tonemap.rs)
- [x] implement parsing CSV files as curves and using them as ior and kappa values. Done, at [parsing.rs](src/parsing.rs)
- [x] implement instances. Somewhat done, still more to do. at [geometry/mod.rs] and [math/transform.rs](src/math/transform.rs)
- [x] implement basic accelerator. Done, at [accelerator.rs](src/accelerator.rs)
- [x] implement environment sampling. Somewhat done, still more to do. at [world.rs](src/world.rs)
- [x] implement light emission sampling to generate rays from lights. Done, part of the material trait at [material.rs](src/material.rs)
- [ ] implement BVH
- [ ] implement spectral power distribution importance sampling. requires computing the CDF of curves.
- [ ] implement scene parser to reduce compilations even more
- [ ] implement light tracing
- [ ] implement BDPT
- [ ] refactor bsdf trait methods

## Contribution

Please view this as a hobby or reference implementation. If you find any issues, please feel free to log them on GitHub's issue tracker, or submit a pull request to fix them :)
