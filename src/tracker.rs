//! The tracking pipeline shared by the preview and headless modes.
//!
//! detect (YuNet) → temporal/largest-face selection → landmark+box smoothing
//! → pose (eyes+nose) → tuning (neutral/gain/deadzone/invert) → One-Euro
//! → UDP to opentrack.

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use ort::environment::Environment;

use crate::face::{Face, FaceDetector};
use crate::filter::OneEuro;
use crate::output::OpenTrack;
use crate::pose::{self, Pose, Tuning};

/// Per-frame output used for drawing and logging.
pub struct Frame {
    pub box_: (f32, f32, f32, f32),
    pub landmarks: [(f32, f32); 5],
    pub eye_dist: f32,
    /// Filtered pose actually sent to opentrack.
    pub pose: Pose,
}

pub struct Tracker {
    detector: FaceDetector,
    udp: OpenTrack,
    pub tuning: Tuning,
    pub neutral: Option<Pose>,
    last_pose: Option<Pose>,
    euro: [OneEuro; 3],
    prev_center: Option<(f32, f32)>,
    smoothed_lm: Option<[(f32, f32); 5]>,
    smoothed_box: Option<(f32, f32, f32, f32)>,
    ref_eye: f32,
    last_time: Instant,
}

impl Tracker {
    /// Load the face detector and open the opentrack UDP output.
    pub fn load(detector_model: &Path, env: &Environment, port: u16) -> Result<Self> {
        Ok(Self {
            detector: FaceDetector::load(detector_model, env)?,
            udp: OpenTrack::connect(port)?,
            tuning: Tuning::default(),
            neutral: None,
            last_pose: None,
            euro: [OneEuro::new(1.0, 0.5, 1.0); 3],
            prev_center: None,
            smoothed_lm: None,
            smoothed_box: None,
            ref_eye: 0.0,
            last_time: Instant::now(),
        })
    }

    /// Capture the current pose as the neutral (zero) reference.
    pub fn recenter(&mut self) {
        if let Some(p) = self.last_pose {
            self.neutral = Some(p);
        }
    }

    /// Process one RGB frame. Returns drawing/pose data, or `None` when no
    /// face is tracked yet.
    pub fn process(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Option<Frame>> {
        let faces = self.detector.detect(rgb, width, height)?;
        if let Some(face) = select_face(&faces, self.prev_center) {
            let eye = face.eye_distance();
            self.ref_eye = (self.ref_eye * 0.999).max(eye);
            // Freeze when the detection shrinks to a sub-region (face lost at
            // extreme yaw).
            if eye >= 0.6 * self.ref_eye {
                let landmarks = match self.smoothed_lm {
                    Some(prev) => blend_landmarks(prev, face.landmarks, self.tuning.smoothing),
                    None => face.landmarks,
                };
                self.smoothed_lm = Some(landmarks);

                let raw_box = (face.x0, face.y0, face.x1, face.y1);
                let box_ = match self.smoothed_box {
                    Some(prev) => blend_box(prev, raw_box, self.tuning.smoothing),
                    None => raw_box,
                };
                self.smoothed_box = Some(box_);
                self.prev_center = Some(face.center());
            }
        }

        let (Some(landmarks), Some(box_)) = (self.smoothed_lm, self.smoothed_box) else {
            return Ok(None);
        };

        let [eye_left, eye_right, nose, _mouth_r, _mouth_l] = landmarks;
        let eye_dist = (eye_right.0 - eye_left.0).hypot(eye_right.1 - eye_left.1);
        let raw_pose = pose::estimate(&[eye_left, eye_right, nose]);
        self.last_pose = Some(raw_pose);

        let shown = self.tuning.apply(raw_pose, self.neutral);
        let now = Instant::now();
        let dt = (now - self.last_time).as_secs_f32();
        self.last_time = now;
        let pose = Pose {
            yaw: self.euro[0].filter(shown.yaw, dt),
            pitch: self.euro[1].filter(shown.pitch, dt),
            roll: self.euro[2].filter(shown.roll, dt),
        };
        let _ = self.udp.send(&pose);

        Ok(Some(Frame {
            box_,
            landmarks,
            eye_dist,
            pose,
        }))
    }
}

/// Pick the largest face (the user's), using the previous centre only as a
/// tie-break. Sub-region false detections are always smaller.
fn select_face<'a>(faces: &'a [Face], prev_center: Option<(f32, f32)>) -> Option<&'a Face> {
    let max_area = faces.iter().map(box_area).fold(0.0_f32, f32::max);
    if max_area <= 0.0 {
        return None;
    }
    let candidates: Vec<&Face> = faces
        .iter()
        .filter(|f| box_area(f) >= 0.8 * max_area)
        .collect();
    let Some((px, py)) = prev_center else {
        return candidates.first().copied();
    };
    candidates.into_iter().min_by(|a, b| {
        let (ax, ay) = a.center();
        let (bx, by) = b.center();
        (ax - px).hypot(ay - py).total_cmp(&(bx - px).hypot(by - py))
    })
}

/// Bounding-box area.
fn box_area(f: &Face) -> f32 {
    (f.x1 - f.x0).max(0.0) * (f.y1 - f.y0).max(0.0)
}

/// Exponential blend of two landmark sets (frame-to-frame smoothing).
fn blend_landmarks(prev: [(f32, f32); 5], cur: [(f32, f32); 5], alpha: f32) -> [(f32, f32); 5] {
    let mut out = cur;
    for (o, (&p, &c)) in out.iter_mut().zip(prev.iter().zip(cur.iter())) {
        *o = (p.0 + (c.0 - p.0) * alpha, p.1 + (c.1 - p.1) * alpha);
    }
    out
}

/// Exponential blend of two bounding boxes `(x0, y0, x1, y1)`.
fn blend_box(
    prev: (f32, f32, f32, f32),
    cur: (f32, f32, f32, f32),
    alpha: f32,
) -> (f32, f32, f32, f32) {
    (
        prev.0 + (cur.0 - prev.0) * alpha,
        prev.1 + (cur.1 - prev.1) * alpha,
        prev.2 + (cur.2 - prev.2) * alpha,
        prev.3 + (cur.3 - prev.3) * alpha,
    )
}
