//! Head pose (yaw, pitch, roll) from the detector's face landmarks.
//!
//! For a game head tracker we only need coarse, monotonic angles — not a full
//! 3D face solve. YuNet gives five points: both eyes, the nose tip and both
//! mouth corners.
//!
//! - `roll`  = angle of the eye line;
//! - `yaw`   = nose offset *along* the eye line, scaled by eye distance;
//! - `pitch` = where the nose sits between the eye line and the mouth line.
//!
//! The nose and mouth offsets are first rotated into the face frame (roll
//! removed), because otherwise a head tilt leaks straight into pitch.
//!
//! Pitch is a *ratio* of two distances measured along the same direction, so
//! it cancels the face's apparent size — which changes a lot with yaw (the
//! eyes foreshorten) and with distance. The nose protrudes, so pitching
//! foreshortens the eye→nose and eye→mouth segments differently, and that
//! ratio moves cleanly with pitch while staying put under yaw and roll.

/// Ratio `nose_down / mouth_down` for a level head, calibrated from a guided
/// capture. Only sets the raw zero point; recentering removes it.
const NOSE_RATIO_REF: f32 = 0.555;
/// Degrees of pitch per unit of ratio, derived from the projected geometry of
/// a typical face (nose ~22 mm proud of the eye plane, mouth ~70 mm below).
const NOSE_RATIO_TO_DEGREES: f32 = 200.0;

/// Head orientation in degrees (coarse, uncalibrated).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    /// Positive = turn right (nose points right of the eye midpoint).
    pub yaw: f32,
    /// Positive = look up.
    pub pitch: f32,
    /// Positive = tilt right (right eye drops in the image).
    pub roll: f32,
}

/// Estimate pose from YuNet's five landmarks in source pixel coordinates.
#[must_use]
pub fn estimate(landmarks: &[(f32, f32); 5]) -> Pose {
    let [eye_left, eye_right, nose, mouth_right, mouth_left] = *landmarks;

    let eye_mid = (
        (eye_left.0 + eye_right.0) * 0.5,
        (eye_left.1 + eye_right.1) * 0.5,
    );
    let mouth_mid = (
        (mouth_right.0 + mouth_left.0) * 0.5,
        (mouth_right.1 + mouth_left.1) * 0.5,
    );
    let eye_vec = (
        eye_right.0 - eye_left.0,
        eye_right.1 - eye_left.1,
    );
    let eye_dist = eye_vec.0.hypot(eye_vec.1).max(1.0);
    let roll = eye_vec.1.atan2(eye_vec.0);

    // Rotate the eye→point offsets into the face frame: `along` runs down the
    // eye line, `down` is perpendicular to it in the face's vertical.
    let (sin, cos) = roll.sin_cos();
    let face = |point: (f32, f32)| {
        let (dx, dy) = (point.0 - eye_mid.0, point.1 - eye_mid.1);
        (dx * cos + dy * sin, -dx * sin + dy * cos)
    };
    let (nose_along, nose_down) = face(nose);
    let (_, mouth_down) = face(mouth_mid);
    let mouth_down = if mouth_down.abs() < 1.0 {
        1.0
    } else {
        mouth_down
    };

    let yaw = nose_along.atan2(eye_dist).to_degrees();
    let ratio = nose_down / mouth_down;
    let pitch = (NOSE_RATIO_REF - ratio) * NOSE_RATIO_TO_DEGREES;

    Pose {
        yaw,
        pitch,
        roll: roll.to_degrees(),
    }
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
    /// The camera sees you mirrored, so turning your head right moves the nose
    /// toward image-left. The camera→head change of basis is a proper 180°
    /// rotation, which flips **yaw and roll** but leaves **pitch** alone.
    /// Inverting only yaw would be a mirror (det = -1): pure yaw and pure roll
    /// still look right, but their combination cross-talks.
    pub invert_yaw: bool,
    pub invert_pitch: bool,
    pub invert_roll: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            yaw_gain: 1.7,
            pitch_gain: 1.2,
            roll_gain: 1.0,
            deadzone: 1.0,
            smoothing: 0.35,
            invert_yaw: true,
            invert_pitch: false,
            invert_roll: true,
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

#[cfg(test)]
mod tests {
    use super::estimate;

    /// Level face with the given nose and mouth heights (pixels below the eyes).
    fn face(nose_y: f32, mouth_y: f32) -> [(f32, f32); 5] {
        [
            (-40.0, 0.0),
            (40.0, 0.0),
            (0.0, nose_y),
            (-28.0, mouth_y),
            (28.0, mouth_y),
        ]
    }

    #[test]
    fn looking_up_raises_pitch() {
        let up = estimate(&face(36.0, 80.0));
        let level = estimate(&face(45.0, 80.0));
        let down = estimate(&face(45.0, 66.0));
        assert!(up.pitch > level.pitch, "{up:?} {level:?}");
        assert!(level.pitch > down.pitch, "{level:?} {down:?}");
    }

    #[test]
    fn head_tilt_does_not_leak_into_pitch() {
        let level = estimate(&face(45.0, 80.0));
        let angle = 20.0_f32.to_radians();
        let (sin, cos) = angle.sin_cos();
        let tilted_landmarks = face(45.0, 80.0).map(|(x, y)| (x * cos - y * sin, x * sin + y * cos));
        let tilted = estimate(&tilted_landmarks);

        assert!(
            (tilted.pitch - level.pitch).abs() < 0.5,
            "pitch moved with roll: {tilted:?} vs {level:?}"
        );
        assert!(
            (tilted.roll - 20.0).abs() < 0.1,
            "roll should read the eye-line angle: {tilted:?}"
        );
    }

    #[test]
    fn yaw_follows_the_nose_sideways() {
        let level = estimate(&face(45.0, 80.0));
        let mut right = face(45.0, 80.0);
        right[2].0 = 30.0;
        assert!(estimate(&right).yaw > level.yaw + 5.0);
    }
}
