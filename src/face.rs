//! YuNet face detector on ONNX Runtime.
//!
//! YuNet (OpenCV Zoo, MIT-licensed) with the multi-scale decode from OpenCV's
//! `FaceDetectorYN`; see `models/NOTICE.md`. The model has a fixed 640x640
//! input, BGR, raw 0..255, and three anchor-free heads (strides 8, 16, 32).

use std::path::Path;

use anyhow::{ensure, Context, Result};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

/// Model input side, in pixels (fixed by the exported model).
pub const INPUT: usize = 640;
/// Feature-map strides and output-head suffixes.
const STRIDES: [usize; 3] = [8, 16, 32];
/// Landmarks per face: right eye, left eye, nose tip, right mouth, left mouth.
pub const N_LANDMARKS: usize = 5;
/// Minimum `sqrt(cls * obj)` score to keep a detection.
const MIN_SCORE: f32 = 0.6;
/// Overlap above which a lower-scoring box is dropped.
const NMS_IOU: f32 = 0.3;

/// A detected face in source pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
    /// Right eye, left eye, nose tip, right mouth corner, left mouth corner.
    pub landmarks: [(f32, f32); N_LANDMARKS],
}

impl Face {
    /// Centre of the bounding box, in source pixels.
    #[must_use]
    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)
    }

    /// Image-left eye, image-right eye and nose tip, in the order the pose
    /// expects. YuNet lists the person's right eye first (image-left).
    #[must_use]
    pub fn pose_points(&self) -> [(f32, f32); 3] {
        [self.landmarks[0], self.landmarks[1], self.landmarks[2]]
    }
}

/// A square crop region for the landmark model, in source pixel coordinates.
#[derive(Clone, Copy, Debug)]
pub struct FaceRect {
    pub cx: f32,
    pub cy: f32,
    pub half: f32,
    /// Rotation (radians) that undoes the eye-line tilt for the crop.
    pub rotation: f32,
}

/// Derive a square face crop from YuNet's landmarks (eyes and mouth corners).
#[must_use]
pub fn face_rect(face: &Face) -> FaceRect {
    const EYE_TO_MOUTH_MIX: f32 = 0.1;
    const EYE_TO_MOUTH_SCALE: f32 = 3.6;
    const EYE_TO_EYE_SCALE: f32 = 4.0;

    let left_eye = face.landmarks[1];
    let right_eye = face.landmarks[0];
    let mouth = (
        (face.landmarks[3].0 + face.landmarks[4].0) * 0.5,
        (face.landmarks[3].1 + face.landmarks[4].1) * 0.5,
    );

    let eye_center = (
        (left_eye.0 + right_eye.0) * 0.5,
        (left_eye.1 + right_eye.1) * 0.5,
    );
    let eye_to_eye = (right_eye.0 - left_eye.0, right_eye.1 - left_eye.1);
    let eye_to_mouth = (mouth.0 - eye_center.0, mouth.1 - eye_center.1);

    let cx = eye_center.0 + eye_to_mouth.0 * EYE_TO_MOUTH_MIX;
    let cy = eye_center.1 + eye_to_mouth.1 * EYE_TO_MOUTH_MIX;

    let eye_to_eye_norm = eye_to_eye.0.hypot(eye_to_eye.1);
    let eye_to_mouth_norm = eye_to_mouth.0.hypot(eye_to_mouth.1);
    let side = (eye_to_mouth_norm * EYE_TO_MOUTH_SCALE).max(eye_to_eye_norm * EYE_TO_EYE_SCALE);
    let rotation = eye_to_eye.1.atan2(eye_to_eye.0);

    FaceRect {
        cx,
        cy,
        half: side * 0.5,
        rotation,
    }
}

pub struct FaceDetector {
    session: Session,
}

impl FaceDetector {
    /// Load the detector from an ONNX file, sharing `env` with other sessions.
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

    /// Detect faces in an interleaved RGB8 frame of `width` x `height` pixels.
    pub fn detect(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<Face>> {
        let (input, scale) = build_input(rgb, width, height)?;
        let tensor = Tensor::from_array(([1_usize, 3, INPUT, INPUT], input))
            .context("failed to build the input tensor")?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => tensor])
            .context("face detector inference failed")?;

        let mut faces = Vec::new();
        for &stride in &STRIDES {
            let cls = output(&outputs, &format!("cls_{stride}"))?;
            let obj = output(&outputs, &format!("obj_{stride}"))?;
            let bbox = output(&outputs, &format!("bbox_{stride}"))?;
            let kps = output(&outputs, &format!("kps_{stride}"))?;
            decode_head(stride, cls, obj, bbox, kps, scale, &mut faces);
        }

        Ok(non_max_suppression(faces))
    }
}

/// Fetch a single f32 output tensor as a slice.
fn output<'a>(outputs: &'a ort::session::SessionOutputs, name: &str) -> Result<&'a [f32]> {
    Ok(outputs
        .get(name)
        .with_context(|| format!("output {name} missing"))?
        .try_extract_tensor::<f32>()
        .with_context(|| format!("output {name} is not a float tensor"))?
        .1)
}

/// Decode one stride's head into `faces` (coordinates in source pixels).
#[expect(
    clippy::indexing_slicing,
    reason = "indices are bounded by the feature-map dimensions derived from the model shapes"
)]
fn decode_head(
    stride: usize,
    cls: &[f32],
    obj: &[f32],
    bbox: &[f32],
    kps: &[f32],
    scale: f32,
    faces: &mut Vec<Face>,
) {
    let grid = INPUT / stride;
    if cls.len() < grid * grid || obj.len() < grid * grid || bbox.len() < grid * grid * 4 || kps.len() < grid * grid * 10 {
        return;
    }
    let stride_f = stride as f32;
    for r in 0..grid {
        for c in 0..grid {
            let idx = r * grid + c;
            let cls_s = cls[idx].clamp(0.0, 1.0);
            let obj_s = obj[idx].clamp(0.0, 1.0);
            let score = (cls_s * obj_s).sqrt();
            if score < MIN_SCORE {
                continue;
            }
            let b = &bbox[idx * 4..idx * 4 + 4];
            let cx = (c as f32 + b[0]) * stride_f;
            let cy = (r as f32 + b[1]) * stride_f;
            let w = b[2].exp() * stride_f;
            let h = b[3].exp() * stride_f;

            let mut landmarks = [(0.0_f32, 0.0_f32); N_LANDMARKS];
            for (n, lm) in landmarks.iter_mut().enumerate() {
                let kx = kps[idx * 10 + 2 * n];
                let ky = kps[idx * 10 + 2 * n + 1];
                *lm = ((c as f32 + kx) * stride_f / scale, (r as f32 + ky) * stride_f / scale);
            }

            faces.push(Face {
                x0: (cx - w * 0.5) / scale,
                y0: (cy - h * 0.5) / scale,
                x1: (cx + w * 0.5) / scale,
                y1: (cy + h * 0.5) / scale,
                score,
                landmarks,
            });
        }
    }
}

/// Letterbox `rgb` into the model's fixed square input as BGR, raw 0..255, NCHW,
/// placed at the top-left (matching OpenCV's `padWithDivisor`). Returns the
/// input tensor and the scale used to map coordinates back to source pixels.
#[expect(
    clippy::indexing_slicing,
    reason = "source and destination offsets are within the validated buffers"
)]
fn build_input(rgb: &[u8], width: usize, height: usize) -> Result<(Vec<f32>, f32)> {
    ensure!(rgb.len() == width * height * 3, "rgb buffer size mismatch");
    let scale = (INPUT as f32 / width as f32).min(INPUT as f32 / height as f32);
    let new_w = ((width as f32 * scale).round() as usize).min(INPUT);
    let new_h = ((height as f32 * scale).round() as usize).min(INPUT);

    let plane = INPUT * INPUT;
    let mut out = vec![0.0_f32; 3 * plane];
    for y in 0..new_h {
        let sy = ((y as f32 + 0.5) / scale).floor() as usize;
        let sy = sy.min(height - 1);
        for x in 0..new_w {
            let sx = ((x as f32 + 0.5) / scale).floor() as usize;
            let sx = sx.min(width - 1);
            let src = (sy * width + sx) * 3;
            let dst = y * INPUT + x;
            // RGB -> BGR, raw 0..255.
            out[dst] = f32::from(rgb[src + 2]);
            out[plane + dst] = f32::from(rgb[src + 1]);
            out[2 * plane + dst] = f32::from(rgb[src]);
        }
    }
    Ok((out, scale))
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
