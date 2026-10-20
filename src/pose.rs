//! Head pose (yaw, pitch, roll) from the detector's face landmarks.
//!
//! For a game head tracker we only need coarse, monotonic angles for generous
//! head turns and bobs — not a full 3D face solve. The eye and nose landmarks
//! are enough:
//!
//! - `roll`  = angle of the eye line (image-right eye relative to image-left);
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

/// Estimate pose from three points in source pixel coordinates: image-left
/// eye, image-right eye, nose tip.
#[must_use]
pub fn estimate(points: &[(f32, f32); 3]) -> Pose {
    let [left_eye, right_eye, nose] = *points;

    let eye_mid_x = (left_eye.0 + right_eye.0) * 0.5;
    let eye_mid_y = (left_eye.1 + right_eye.1) * 0.5;
    let eye_dist = (right_eye.0 - left_eye.0)
        .hypot(right_eye.1 - left_eye.1)
        .max(1.0);

    let yaw = (nose.0 - eye_mid_x).atan2(eye_dist).to_degrees();
    let pitch = (eye_mid_y - nose.1).atan2(eye_dist).to_degrees();
    let roll = (right_eye.1 - left_eye.1)
        .atan2(right_eye.0 - left_eye.0)
        .to_degrees();

    Pose { yaw, pitch, roll }
}

/// Runtime pose tuning: neutral subtraction, gain, deadzone, inversion.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    /// Multiplier on yaw (1.0 = raw angle).
    pub yaw_gain: f32,
    /// Multiplier on pitch.
    pub pitch_gain: f32,
    /// Multiplier on roll.
    pub roll_gain: f32,
    /// Degrees ignored around neutral (kills jitter).
    pub deadzone: f32,
    /// EMA smoothing factor (0..1; lower = smoother/slower).
    pub smoothing: f32,
    pub invert_yaw: bool,
    pub invert_pitch: bool,
    pub invert_roll: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            yaw_gain: 1.7,
            pitch_gain: 2.0,
            roll_gain: 1.0,
            deadzone: 0.5,
            smoothing: 0.35,
            invert_yaw: true,
            invert_pitch: false,
            invert_roll: false,
        }
    }
}

impl Tuning {
    /// Turn a raw smoothed pose into the output pose: subtract neutral, apply
    /// gain, deadzone and inversion.
    #[must_use]
    pub fn apply(&self, pose: Pose, neutral: Option<Pose>) -> Pose {
        let n = neutral.unwrap_or_default();
        // Deadzone is applied in head-degrees (before gain), so it absorbs
        // landmark noise around the neutral pose; gain then amplifies intent.
        let mut yaw = apply_deadzone(pose.yaw - n.yaw, self.deadzone) * self.yaw_gain;
        let mut pitch = apply_deadzone(pose.pitch - n.pitch, self.deadzone) * self.pitch_gain;
        let mut roll = apply_deadzone(pose.roll - n.roll, self.deadzone) * self.roll_gain;
        if self.invert_yaw {
            yaw = -yaw;
        }
        if self.invert_pitch {
            pitch = -pitch;
        }
        if self.invert_roll {
            roll = -roll;
        }
        Pose { yaw, pitch, roll }
    }
}

/// Zero out values inside `deadzone`, then subtract it so output is continuous.
fn apply_deadzone(value: f32, deadzone: f32) -> f32 {
    if value.abs() <= deadzone {
        0.0
    } else {
        value - deadzone * value.signum()
    }
}
