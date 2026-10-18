//! Head pose (yaw, pitch, roll) from the BlazeFace detection keypoints.
//!
//! For a game head tracker we only need coarse, monotonic angles for generous
//! head turns and bobs — not a full 3D face solve. The detector's six
//! keypoints (left eye, right eye, nose, mouth, ears) are enough:
//!
//! - `roll`  = angle of the eye line (right eye relative to left eye);
//! - `yaw`   = nose offset from the eye midpoint, scaled by eye distance;
//! - `pitch` = nose height below the eye line, scaled by eye distance.

/// Head orientation in degrees (coarse, uncalibrated).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    /// Positive = turn right (nose points right of the eye midpoint).
    pub yaw: f32,
    /// Positive = look up. Biased by the natural nose offset; calibrate later.
    pub pitch: f32,
    /// Positive = tilt right (right eye drops in the image).
    pub roll: f32,
}

/// Estimate pose from three points in source pixel coordinates: left eye,
/// right eye, nose tip.
///
/// Keypoint order: left eye, right eye, nose tip.
#[must_use]
pub fn estimate(points: &[(f32, f32); 3]) -> Pose {
    let [left_eye, right_eye, nose] = *points;

    let eye_mid_x = (left_eye.0 + right_eye.0) * 0.5;
    let eye_mid_y = (left_eye.1 + right_eye.1) * 0.5;
    let eye_dist = (right_eye.0 - left_eye.0).hypot(right_eye.1 - left_eye.1).max(1.0);

    let yaw = (nose.0 - eye_mid_x).atan2(eye_dist).to_degrees();
    let pitch = (eye_mid_y - nose.1).atan2(eye_dist).to_degrees();
    let roll = (right_eye.1 - left_eye.1).atan2(right_eye.0 - left_eye.0).to_degrees();

    Pose { yaw, pitch, roll }
}
