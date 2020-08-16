use packed_simd::f32x4;

use std::cmp::PartialEq;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LensType {
    Solid,
    Air,
    Aperture,
}

#[derive(Copy, Clone, Debug)]
pub struct LensInterface {
    pub radius: f32,
    pub thickness_short: f32,
    pub thickness_mid: f32,
    pub thickness_long: f32,
    pub anamorphic: bool,
    pub lens_type: LensType,
    pub ior: f32, // index of refraction
    pub vno: f32, // abbe number
    pub housing_radius: f32,
    pub aspheric: i32,
    pub correction: f32x4,
}

impl LensInterface {
    pub fn thickness_at(self, mut zoom: f32) -> f32 {
        if zoom < 0.5 {
            zoom *= 2.0;
            self.thickness_short * (1.0 - zoom) + self.thickness_mid * zoom
        } else {
            zoom -= 0.5;
            zoom *= 2.0;
            self.thickness_mid * (1.0 - zoom) + self.thickness_long * zoom
        }
    }

    pub fn parse_from(string: &str, default_ior: f32, default_vno: f32) -> Result<Self, &str> {
        // format is:
        // lens := radius thickness_short(/thickness_mid(/thickness_long)?)? (anamorphic)? (mtl_name|'air'|'iris') ior vno housing_radius ('#!aspheric='aspheric_correction)?
        // radius := float
        // thickness_short := float
        // thickness_mid := float
        // thickness_long := float
        // anamorphic := 'cx_'
        // mtl_name := word
        // ior := float
        // vno := float
        // housing_radius := float
        // aspheric_correction := (float','){3}float

        if string.starts_with("#") {
            return Err("line started with comment");
        }
        println!("{}", string);
        let mut tokens = string.split_ascii_whitespace();
        let radius = tokens
            .next()
            .ok_or("ran out of tokens at radius")?
            .parse::<f32>()
            .map_err(|e| "err parsing float at radius")?;
        let thickness_token: &str = tokens
            .next()
            .ok_or("ran out of tokens at thickness token")?;
        let mut thickness_iterator = thickness_token.split("/");
        let thickness_short = thickness_iterator
            .next()
            .unwrap()
            .parse::<f32>()
            .map_err(|e| "err parsing float at thickness short")?;
        let thickness_mid = match thickness_iterator.next() {
            Some(token) => token
                .parse::<f32>()
                .map_err(|e| "err parsing float at thickness mid")?,
            None => thickness_short,
        };
        let thickness_long = match thickness_iterator.next() {
            Some(token) => token
                .parse::<f32>()
                .map_err(|e| "err parsing float at thickness long")?,
            None => thickness_short,
        };
        let maybe_anamorphic_or_lens = tokens.next().ok_or("ran out of tokens at anamorphic")?;
        let anamorphic = maybe_anamorphic_or_lens == "cx_";
        let next_token = if !anamorphic {
            maybe_anamorphic_or_lens
        } else {
            tokens.next().ok_or("ran out of tokens at lens type")?
        };
        let lens_type = match next_token {
            "air" => LensType::Air,
            "iris" => LensType::Aperture,
            _ => LensType::Solid,
        };
        let (ior, vno, housing_radius);
        let (a, b) = (tokens.next(), tokens.next());
        match (a, b) {
            (Some(token1), Some(token2)) => {
                ior = token1
                    .parse::<f32>()
                    .map_err(|e| "err parsing float at ior")?;
                vno = token2
                    .parse::<f32>()
                    .map_err(|e| "err parsing float at vno")?;
                housing_radius = tokens
                    .next()
                    .ok_or("ran out of tokens at housing radius branch 1")?
                    .parse::<f32>()
                    .map_err(|e| "err parsing float at housing radius branch 1")?;
                let _aspheric = tokens.next();
            }
            (Some(token1), None) => {
                // this must be the situation where there is a housing radius but no aspheric correction.
                ior = match lens_type {
                    LensType::Solid => default_ior,
                    _ => 1.0,
                };
                vno = match lens_type {
                    LensType::Solid => default_vno,
                    _ => 0.0,
                };
                housing_radius = token1
                    .parse::<f32>()
                    .map_err(|e| "error parsing float at housing radius branch 2")?;
            }
            (None, None) => {
                return Err("ran_out_of_tokens");
            }
            (None, Some(token1)) => {
                return Err("what the fuck");
            }
        }

        Ok(LensInterface {
            radius,
            thickness_short,
            thickness_mid,
            thickness_long,
            anamorphic,
            lens_type,
            ior,
            vno,
            housing_radius,
            aspheric: 0,
            correction: f32x4::splat(0.0),
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_parse() {
        let test_string = "65.22 9.60  N-SSK8 1.5 50 24.0";
        let lens = LensInterface::parse_from(test_string, 1.0, 0.0);
        println!("{:?}", lens);
    }
}
