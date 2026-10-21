//! The tracking pipeline shared by the preview and headless modes.
//!
//! detect (YuNet) → temporal/largest-face selection → landmark+box smoothing
//! → pose (eyes, nose and mouth) → tuning (neutral/gain/deadzone/invert)
//! → One-Euro → UDP to opentrack.

use std::env::var_os;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use nalgebra::Vector3;
use ort::environment::Environment;

use crate::face::{self, Face, FaceDetector};
use crate::filter::OneEuro;
use crate::landmark::FaceMesh;
use crate::output::OpenTrack;
use crate::pose::{self, Pose, Tuning};
use crate::pose3d;

/// Physical rig, used to reconcile "look at the camera" with "look at the
/// screen".
///
/// The camera sits on top of the monitor, so gazing at the camera is a few
/// degrees above gazing at the screen centre. Recentring on the camera without
/// correcting for that leaves the neutral high, and every normal glance at the
/// screen reads as negative pitch.
mod rig {
    /// Monitor diagonal, inches.
    pub const MONITOR_DIAGONAL_IN: f32 = 32.0;
    /// Panel aspect: width, height.
    pub const MONITOR_ASPECT: (f32, f32) = (16.0, 9.0);
    /// Eyes to the screen surface, cm (measured to the screen centre).
    pub const EYE_TO_SCREEN_CM: f32 = 66.0;
    /// Camera centre above the top edge of the panel, cm.
    pub const CAMERA_ABOVE_EDGE_CM: f32 = 1.0;
    /// Eyes above the screen centre, cm (0 = level with it).
    ///
    /// Cross-check: with these numbers the camera ends up 10.9 cm above the
    /// eyes, so its distance from the eyes is sqrt(66² + 10.9²) = 66.9 cm,
    /// matching the measured 67 cm.
    pub const EYE_ABOVE_CENTRE_CM: f32 = 10.0;

    /// How far below the camera the screen centre sits, in degrees of pitch.
    #[must_use]
    pub fn screen_offset_deg() -> f32 {
        let (width, height) = MONITOR_ASPECT;
        let panel_cm = MONITOR_DIAGONAL_IN * 2.54 * height / (width * width + height * height).sqrt();
        let camera_above_centre = panel_cm / 2.0 + CAMERA_ABOVE_EDGE_CM;
        let rise = camera_above_centre - EYE_ABOVE_CENTRE_CM;
        rise.atan2(EYE_TO_SCREEN_CM).to_degrees()
    }
}

/// Consecutive sub-region rejections after which the detection is accepted
/// anyway (about half a second at 30 fps), so a real change in face size
/// cannot freeze the tracker.
const FREEZE_LIMIT: u32 = 15;
/// Consecutive rejections after which even an implausible jump is accepted, so
/// a frozen tracker can re-acquire a face that moved far away. Long enough
/// (1.5 s at 30 fps) that brief false positives never get through.
const REACQUIRE_LIMIT: u32 = 45;
/// Smallest believable eye separation, in pixels. A fragment of the face, or
/// whatever the detector finds when the camera is covered, is far below this;
/// at a normal desk distance a real face is ~60-90 px. No delay or bypass may
/// accept anything smaller, otherwise the baseline collapses and the tracker
/// believes it is still looking at a face.
const MIN_FACE_EYE_PX: f32 = 25.0;
/// Largest believable yaw/pitch/roll change between two frames. At 30 fps this
/// is 1200°/s, far beyond any real head motion, so it only rejects garbage
/// detections (fragments of the face, glasses highlights) that would otherwise
/// teleport the pose.
const MAX_JUMP_DEG: f32 = 40.0;

/// Per-frame output used for drawing and logging.
#[derive(Clone, Copy)]
pub struct Frame {
    pub box_: (f32, f32, f32, f32),
    pub landmarks: [(f32, f32); 5],
    pub eye_dist: f32,
    /// Filtered pose actually sent to opentrack.
    pub pose: Pose,
    /// Pose straight from the landmarks, before neutral/gain/deadzone.
    pub raw_pose: Pose,
    /// Whether this frame had a fresh detection (false = held/stale pose).
    pub detected: bool,
}

pub struct Tracker {
    detector: FaceDetector,
    mesh: FaceMesh,
    udp: OpenTrack,
    pub tuning: Tuning,
    pub neutral: Option<Pose>,
    last_pose: Option<Pose>,
    euro: [OneEuro; 3],
    prev_center: Option<(f32, f32)>,
    smoothed_lm: Option<[(f32, f32); 5]>,
    smoothed_box: Option<(f32, f32, f32, f32)>,
    ref_eye: f32,
    /// Consecutive frames rejected by the sub-region guard.
    frozen_frames: u32,
    /// Pitch bias applied at recenter: the screen centre is this many degrees
    /// below the camera, so zero means "looking at the screen", not the lens.
    screen_offset: f32,
    /// 3D landmark cloud captured at recenter (the pose zero point).
    reference: Option<Vec<Vector3<f32>>>,
    last_cloud: Option<Vec<Vector3<f32>>>,
    /// Opt-in 3D pose (`HEADTRACK_3D=1`, `--3d`, or `3` in preview); still
    /// being stabilised.
    pub use_3d: bool,
    last_time: Instant,
}

impl Tracker {
    /// Load the face detector, the landmark model and the opentrack UDP output.
    pub fn load(
        detector_model: &Path,
        landmark_model: &Path,
        env: &Environment,
        port: u16,
    ) -> Result<Self> {
        let screen_offset = rig::screen_offset_deg();
        log::info!(
            "screen centre sits {screen_offset:.1}° below the camera; \
             recenter while looking at the camera"
        );
        Ok(Self {
            detector: FaceDetector::load(detector_model, env)?,
            mesh: FaceMesh::load(landmark_model, env)?,
            udp: OpenTrack::connect(port)?,
            tuning: Tuning::default(),
            neutral: None,
            last_pose: None,
            euro: [OneEuro::new(1.0, 0.5, 1.0); 3],
            prev_center: None,
            smoothed_lm: None,
            smoothed_box: None,
            ref_eye: 0.0,
            frozen_frames: 0,
            screen_offset,
            reference: None,
            last_cloud: None,
            use_3d: var_os("HEADTRACK_3D").is_some(),
            last_time: Instant::now(),
        })
    }

    /// Capture the current pose as the neutral (zero) reference.
    ///
    /// The user looks at the camera to recenter, so the pitch is biased down by
    /// the camera's height above the screen: zero then means looking at the
    /// screen centre, which is where the eyes rest in normal use.
    pub fn recenter(&mut self) {
        self.reference = self.last_cloud.clone();
        if let Some(mut pose) = self.last_pose {
            pose.pitch -= self.screen_offset;
            self.neutral = Some(pose);
        }
    }

    /// Process one RGB frame. Returns drawing/pose data, or `None` when no
    /// face is tracked yet.
    pub fn process(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Option<Frame>> {
        let faces = self.detector.detect(rgb, width, height)?;
        let mut detected = false;
        if let Some(face) = select_face(&faces, self.prev_center) {
            let eye = face.eye_distance();
            let candidate = pose::estimate(&face.landmarks);
            // Reject a sudden collapse to a sub-region (face lost at extreme
            // yaw). A genuine turn foreshortens the eyes gradually, while a
            // sub-region detection drops by more than half in one step.
            //
            // The reference is the last *accepted* distance, not the all-time
            // maximum: the maximum could only decay 0.1% per frame, so a turn
            // that pushed past 60% of it froze the tracker until the user
            // turned back. A long run of rejections means the face really did
            // shrink, so accept and re-baseline then.
            let plausible = self.ref_eye <= 0.0 || eye >= 0.6 * self.ref_eye;
            // A fragment of the face can pass the size check briefly; its pose
            // is nowhere near the current one, so gate on head-motion too.
            let steady = self.last_pose.is_none_or(|prev| {
                (candidate.yaw - prev.yaw).abs() <= MAX_JUMP_DEG
                    && (candidate.pitch - prev.pitch).abs() <= MAX_JUMP_DEG
                    && (candidate.roll - prev.roll).abs() <= MAX_JUMP_DEG
            });
            // Hard floor: never accept a "face" smaller than MIN_FACE_EYE_PX,
            // not even via the freeze or re-acquire bypasses.
            let big_enough = eye >= MIN_FACE_EYE_PX;
            let accept = big_enough
                && (((plausible || self.frozen_frames >= FREEZE_LIMIT) && steady)
                    || self.frozen_frames >= REACQUIRE_LIMIT);
            if accept {
                self.ref_eye = eye;
                self.frozen_frames = 0;
                detected = true;
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
            } else {
                self.frozen_frames += 1;
            }
        }

        let (Some(landmarks), Some(box_)) = (self.smoothed_lm, self.smoothed_box) else {
            return Ok(None);
        };

        let [eye_left, eye_right, _, _, _] = landmarks;
        let eye_dist = (eye_right.0 - eye_left.0).hypot(eye_right.1 - eye_left.1);

        // 3D pose (opt-in, still being stabilised): run FaceMesh on the
        // smoothed face crop and align its landmark cloud to the reference
        // captured at recenter.
        let pose_3d = if self.use_3d {
            let crop_face = Face {
                x0: box_.0,
                y0: box_.1,
                x1: box_.2,
                y1: box_.3,
                score: 1.0,
                landmarks,
            };
            let mut rect = face::face_rect(&crop_face);
            // Feed FaceMesh an axis-aligned crop: the eye-alignment rotation
            // would otherwise remove roll and add per-frame rotation noise.
            rect.rotation = 0.0;
            let (mesh_landmarks, mesh_score) = self.mesh.run(rgb, width, height, &rect)?;
            let cloud = pose3d::cloud(&mesh_landmarks);
            if mesh_score > 0.0 && cloud.len() >= 3 {
                self.last_cloud = Some(cloud.clone());
                if self.reference.is_none() {
                    self.reference = Some(cloud.clone());
                }
                self.reference.as_ref().and_then(|reference| {
                    pose3d::rotation(&cloud, reference).map(|r| {
                        let (yaw, pitch, roll) = pose3d::angles(&r);
                        Pose { yaw, pitch, roll }
                    })
                })
            } else {
                None
            }
        } else {
            None
        };

        let raw_pose = pose_3d.unwrap_or_else(|| pose::estimate(&landmarks));
        self.last_pose = Some(raw_pose);

        // The 3D rotation is already relative to the recenter reference, so it
        // must not subtract `neutral` again; the 2D fallback still needs it.
        let neutral = if pose_3d.is_some() { None } else { self.neutral };
        let shown = self.tuning.apply(raw_pose, neutral);
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
            raw_pose,
            detected,
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
