//! FaceMesh 468-point facial landmark inference on ONNX Runtime.
//!
//! The model is MediaPipe's FaceMesh 192x192 (Apache-2.0), redistributed by
//! PINTO's model zoo; see `models/NOTICE.md`. Input is NCHW normalised to
//! `[-1, 1]`; output is 468 landmarks (x, y, z) in 192-pixel input space plus
//! a raw score logit.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::face::FaceRect;

/// Square network input side, in pixels.
pub const INPUT: usize = 192;
/// Number of landmarks the model predicts.
pub const N_LANDMARKS: usize = 468;

/// One facial landmark in the 192x192 network-input coordinate space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub struct FaceMesh {
    session: Session,
}

impl FaceMesh {
    /// Load the landmark model, sharing `env` with other sessions.
    pub fn load(model: &Path, env: &Environment) -> Result<Self> {
        let mut builder =
            Session::builder(env).context("failed to create ONNX Runtime session builder")?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|err| anyhow::anyhow!("failed to set graph optimization level: {err}"))?;
        let session = builder
            .commit_from_file(model)
            .with_context(|| format!("failed to load {}", model.display()))?;
        Ok(Self { session })
    }

    /// Run landmark inference on a square crop of `rgb` described by `rect`.
    ///
    /// Returns the 468 landmarks (in 192-space) and the raw score logit.
    pub fn run(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        rect: &FaceRect,
    ) -> Result<(Vec<Landmark>, f32)> {
        let input = crop_nchw(rgb, width, height, rect);
        let tensor = Tensor::from_array(([1_usize, 3, INPUT, INPUT], input))
            .context("failed to build the input tensor")?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => tensor])
            .context("landmark inference failed")?;

        let (_, raw) = outputs
            .get("landmarks")
            .context("landmarks output missing")?
            .try_extract_tensor::<f32>()
            .context("landmarks output is not a float tensor")?;
        ensure!(
            raw.len() == N_LANDMARKS * 3,
            "landmarks have {} values",
            raw.len()
        );

        let landmarks = raw
            .chunks_exact(3)
            .map(|c| {
                let [x, y, z] = c else {
                    unreachable!("chunks_exact(3) yields three-element slices");
                };
                Landmark {
                    x: *x,
                    y: *y,
                    z: *z,
                }
            })
            .collect();

        let score = outputs
            .get("score")
            .context("score output missing")?
            .try_extract_tensor::<f32>()
            .context("score output is not a float tensor")?
            .1
            .first()
            .copied()
            .unwrap_or(f32::NEG_INFINITY);

        Ok((landmarks, score))
    }

    /// Map a 192-space landmark coordinate back to source pixel coordinates.
    #[must_use]
    pub fn to_source(&self, x: f32, y: f32, rect: &FaceRect) -> (f32, f32) {
        let (sin, cos) = rect.rotation.sin_cos();
        let ux = x / INPUT as f32 * 2.0 - 1.0;
        let uy = y / INPUT as f32 * 2.0 - 1.0;
        let rx = ux * cos - uy * sin;
        let ry = ux * sin + uy * cos;
        (rect.cx + rx * rect.half, rect.cy + ry * rect.half)
    }
}

/// Crop the square `rect` region (rotated to align the eye line), resize to
/// `INPUT` x `INPUT` with bilinear sampling, and normalise to `[-1, 1]` NCHW.
#[expect(
    clippy::indexing_slicing,
    reason = "output plane offsets are within the fixed INPUT*INPUT buffer"
)]
fn crop_nchw(rgb: &[u8], width: usize, height: usize, rect: &FaceRect) -> Vec<f32> {
    let plane = INPUT * INPUT;
    let mut out = vec![0.0_f32; 3 * plane];
    let (sin, cos) = rect.rotation.sin_cos();
    let half = rect.half;
    let norm = 2.0 / INPUT as f32;

    for oy in 0..INPUT {
        let uy = (oy as f32 + 0.5) * norm - 1.0;
        for ox in 0..INPUT {
            let ux = (ox as f32 + 0.5) * norm - 1.0;
            let rx = ux * cos - uy * sin;
            let ry = ux * sin + uy * cos;
            let sx = rect.cx + rx * half;
            let sy = rect.cy + ry * half;
            let (r, g, b) = sample_bilinear(rgb, width, height, sx, sy);
            let dst = oy * INPUT + ox;
            out[dst] = r / 127.5 - 1.0;
            out[plane + dst] = g / 127.5 - 1.0;
            out[2 * plane + dst] = b / 127.5 - 1.0;
        }
    }
    out
}

/// Bilinear sample of an interleaved RGB8 buffer at `(x, y)`, clamped to the
/// image edges.
#[expect(
    clippy::indexing_slicing,
    reason = "sample coordinates are clamped into the validated frame"
)]
fn sample_bilinear(rgb: &[u8], width: usize, height: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let x = x.clamp(0.0, width as f32 - 1.0);
    let y = y.clamp(0.0, height as f32 - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(width as f32 - 1.0);
    let y1 = (y0 + 1.0).min(height as f32 - 1.0);
    let fx = x - x0;
    let fy = y - y0;

    let (x0, x1) = (x0 as usize, x1 as usize);
    let (y0, y1) = (y0 as usize, y1 as usize);

    let mut acc = [0.0_f32; 3];
    for (c, a) in acc.iter_mut().enumerate() {
        let p00 = f32::from(rgb[(y0 * width + x0) * 3 + c]);
        let p10 = f32::from(rgb[(y0 * width + x1) * 3 + c]);
        let p01 = f32::from(rgb[(y1 * width + x0) * 3 + c]);
        let p11 = f32::from(rgb[(y1 * width + x1) * 3 + c]);
        let top = p00 + (p10 - p00) * fx;
        let bottom = p01 + (p11 - p01) * fx;
        *a = top + (bottom - top) * fy;
    }
    let [r, g, b] = acc;
    (r, g, b)
}
