# Rust Pathtracer

This is a Single Wavelength Physically based 3D rendering engine written in Rust. Performance is not the focus, as the focus is mostly in implementing concepts in a concise and readable manner.

The purpose is to help me become more familiar with Rust and with general Light Transport algorithms. However if it helps the reader learn more about these concepts, that would be great.

It supports the following integrators:
* [Path Tracing](src/integrator/pt.rs)

Work is in progress on other branches for the following integrators:
* [Light tracing](src/integrator/lt.rs)
* [Bidirectional Path Tracing](src/integrator/bdpt/mod.rs)

In addition, much of the code emphasizes matching reality as closely as possible, including the following:
* Output format:
  * The renderer outputs in come in a .exr file in Linear RGB space, and a .png file in sRGB space.
    * custom exposure values are supported, however they are not part of the config file as of yet. the default behavior is to set the brightest pixel on the screen to white.
* Colors and Lights:
  * Colors on the film are represented in [CIE XYZ](https://en.wikipedia.org/wiki/CIE_1931_color_space) color space. this is then color mapped to linear RGB space and sRGB space according to the [wikipedia article](https://en.wikipedia.org/wiki/SRGB)
  * Lights can have physically correct spectral power distribution functions.
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

The Light Tracing Integrator and the Bidirectional Path tracing integrator can represent the camera's lens in the scene, and allow for light to intersect the camera lens. While typically this is done with one camera at a time, I've taken the liberty of trying to implement it so that multiple cameras can be in the scene at once, and if light happens to intersect *any* camera, some contribution will be recorded. This is divided so that all the films that use the Light Tracing integrator will all have their cameras in the scene at once, and the same for Bidirectional Path Tracing.

That should theoretically cause images to converge faster, though the feature is still a WIP and may be changed in the future. Renders using the Path tracing integrator will be unaffected, and their cameras will not physically exist in the World.

Please view this as a hobby or reference implementation, and if you find any issues, please feel free to submit a pull request to fix them, or log them on GitHub's issue tracker :)
