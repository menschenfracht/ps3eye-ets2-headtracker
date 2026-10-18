//! BlazeFace short-range face detector on ONNX Runtime.
//!
//! The model and its anchor layout come from MediaPipe's face detection
//! graph (Apache-2.0); see `models/NOTICE.md`.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// Square network input side, in pixels.
pub const INPUT: usize = 128;
/// Number of anchors the model predicts per image.
const ANCHORS: usize = 896;
/// Values per anchor: 4 box numbers followed by 6 keypoints (x, y each).
const COORDS: usize = 16;
/// Minimum sigmoid score to keep a detection.
const MIN_SCORE: f32 = 0.5;
/// Overlap above which a lower-scoring box is dropped.
const NMS_IOU: f32 = 0.3;

/// A face found in the letterboxed network input, normalised to `0..=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
    /// Right eye, left eye, mouth centre, nose tip, left ear, right ear.
    pub keypoints: [(f32, f32); 6],
}

/// Letterbox mapping from the source frame to the network input.
#[derive(Clone, Copy, Debug)]
pub struct Letterbox {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

impl Letterbox {
    /// Map a normalised network-input coordinate back to source pixels.
    #[must_use]
    pub fn to_source(&self, x: f32, y: f32) -> (f32, f32) {
        let side = INPUT as f32;
        ((x * side - self.pad_x) / self.scale, (y * side - self.pad_y) / self.scale)
    }
}

/// One square anchor: centre (normalised) and side length (normalised).
#[derive(Clone, Copy, Debug)]
struct Anchor {
    cx: f32,
    cy: f32,
    size: f32,
}

pub struct FaceDetector {
    session: Session,
    anchors: Vec<Anchor>,
}

impl FaceDetector {
    /// Load the detector from an ONNX file, sharing `env` with other sessions.
    ///
    /// `ort` allows a single [`Environment`] per process, so callers must build
    /// one once and pass it here.
    pub fn load(model: &Path, env: &Environment) -> Result<Self> {
        let mut builder = Session::builder(env)
            .context("failed to create ONNX Runtime session builder")?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|err| anyhow::anyhow!("failed to set graph optimization level: {err}"))?;
        let session = builder
            .commit_from_file(model)
            .with_context(|| format!("failed to load {}", model.display()))?;
        let anchors = generate_anchors();
        ensure!(anchors.len() == ANCHORS, "anchor count {} != {ANCHORS}", anchors.len());
        Ok(Self { session, anchors })
    }

    /// Detect faces in an interleaved RGB8 frame of `width` x `height` pixels.
    pub fn detect(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<(Vec<Face>, Letterbox)> {
        let (input, letterbox) = letterbox_nchw(rgb, width, height)?;
        let tensor = Tensor::from_array(([1_usize, 3, INPUT, INPUT], input))
            .context("failed to build the input tensor")?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => tensor])
            .context("face detector inference failed")?;

        let (_, regressors) = outputs
            .get("regressors")
            .context("regressors output missing")?
            .try_extract_tensor::<f32>()
            .context("regressors output is not a float tensor")?;
        let (_, scores) = outputs
            .get("classificators")
            .context("classificators output missing")?
            .try_extract_tensor::<f32>()
            .context("classificators output is not a float tensor")?;
        ensure!(regressors.len() == ANCHORS * COORDS, "regressors have {} values", regressors.len());
        ensure!(scores.len() == ANCHORS, "scores have {} values", scores.len());

        let faces = decode(regressors, scores, &self.anchors);
        Ok((non_max_suppression(faces), letterbox))
    }
}

/// Resize to fit the square input (keeping aspect), pad with black, and
/// normalise to `[-1, 1]` in NCHW order.
#[expect(
    clippy::indexing_slicing,
    reason = "pixel offsets are within the validated frame and plane buffers"
)]
fn letterbox_nchw(rgb: &[u8], width: usize, height: usize) -> Result<(Vec<f32>, Letterbox)> {
    ensure!(rgb.len() == width * height * 3, "rgb buffer size mismatch");
    let side = INPUT as f32;
    let long = width.max(height) as f32;
    let scale = side / long;
    let new_w = (width as f32 * scale).round() as usize;
    let new_h = (height as f32 * scale).round() as usize;
    let pad_x = ((INPUT - new_w) / 2) as f32;
    let pad_y = ((INPUT - new_h) / 2) as f32;
    let letterbox = Letterbox { scale, pad_x, pad_y };

    let plane = INPUT * INPUT;
    let mut out = vec![-1.0_f32; 3 * plane];
    let pad_x = pad_x as usize;
    let pad_y = pad_y as usize;
    for y in 0..new_h {
        let sy = ((y as f32 + 0.5) / scale).floor() as usize;
        let sy = sy.min(height - 1);
        for x in 0..new_w {
            let sx = ((x as f32 + 0.5) / scale).floor() as usize;
            let sx = sx.min(width - 1);
            let src = (sy * width + sx) * 3;
            let dst = (pad_y + y) * INPUT + pad_x + x;
            for c in 0..3 {
                let v = f32::from(rgb[src + c]) / 127.5 - 1.0;
                out[c * plane + dst] = v;
            }
        }
    }
    Ok((out, letterbox))
}

/// MediaPipe SSD anchors for the short-range model: strides 8, 16, 16, 16,
/// two square anchors per cell (scale and interpolated scale), row-major.
#[expect(
    clippy::indexing_slicing,
    reason = "layer and layer + 1 are both within the scales table by construction"
)]
fn generate_anchors() -> Vec<Anchor> {
    const STRIDES: [usize; 4] = [8, 16, 16, 16];
    const MIN_SCALE: f32 = 0.148_437_5;
    const MAX_SCALE: f32 = 0.75;
    let layers = STRIDES.len();

    let mut scales: Vec<f32> = (0..layers)
        .map(|i| MIN_SCALE + (MAX_SCALE - MIN_SCALE) * i as f32 / (layers - 1) as f32)
        .collect();
    scales.push(1.0);

    let mut anchors = Vec::with_capacity(ANCHORS);
    for (layer, &stride) in STRIDES.iter().enumerate() {
        let grid = INPUT.div_ceil(stride);
        let scale = scales[layer];
        let interpolated = (scale * scales[layer + 1]).sqrt();
        for y in 0..grid {
            for x in 0..grid {
                let cx = (x as f32 + 0.5) / grid as f32;
                let cy = (y as f32 + 0.5) / grid as f32;
                anchors.push(Anchor { cx, cy, size: scale });
                anchors.push(Anchor { cx, cy, size: interpolated });
            }
        }
    }
    anchors
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-100.0, 100.0)).exp())
}

#[expect(
    clippy::indexing_slicing,
    reason = "score index matches anchors; regressor slice is a fixed COORDS-length window"
)]
fn decode(regressors: &[f32], scores: &[f32], anchors: &[Anchor]) -> Vec<Face> {
    let side = INPUT as f32;
    let mut faces = Vec::new();
    for (i, anchor) in anchors.iter().enumerate() {
        let score = sigmoid(scores[i]);
        if score < MIN_SCORE {
            continue;
        }
        let Some(r) = regressors.get(i * COORDS..(i + 1) * COORDS) else {
            continue;
        };
        let cx = r[0] / side * anchor.size + anchor.cx;
        let cy = r[1] / side * anchor.size + anchor.cy;
        let w = r[2] / side * anchor.size;
        let h = r[3] / side * anchor.size;
        let mut keypoints = [(0.0_f32, 0.0_f32); 6];
        for (k, point) in keypoints.iter_mut().enumerate() {
            let px = r[4 + 2 * k] / side * anchor.size + anchor.cx;
            let py = r[5 + 2 * k] / side * anchor.size + anchor.cy;
            *point = (px, py);
        }
        faces.push(Face {
            x0: cx - w / 2.0,
            y0: cy - h / 2.0,
            x1: cx + w / 2.0,
            y1: cy + h / 2.0,
            score,
            keypoints,
        });
    }
    faces
}

fn iou(a: &Face, b: &Face) -> f32 {
    let ix = (a.x1.min(b.x1) - a.x0.max(b.x0)).max(0.0);
    let iy = (a.y1.min(b.y1) - a.y0.max(b.y0)).max(0.0);
    let inter = ix * iy;
    let area_a = (a.x1 - a.x0).max(0.0) * (a.y1 - a.y0).max(0.0);
    let area_b = (b.x1 - b.x0).max(0.0) * (b.y1 - b.y0).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn non_max_suppression(mut faces: Vec<Face>) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Face> = Vec::new();
    for face in faces {
        if kept.iter().all(|k| iou(k, &face) < NMS_IOU) {
            kept.push(face);
        }
    }
    kept
}
