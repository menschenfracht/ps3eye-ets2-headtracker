//! 3D head pose from FaceMesh landmarks via Kabsch alignment.
//!
//! Instead of coupled 2D angles, we align the current 3D landmark cloud to a
//! reference cloud captured at recenter and take the rotation between them.
//! Landmark indices are chosen from stable regions (forehead, eyes, nose,
//! cheeks, chin) and exclude the lips and lower jaw so talking does not move
//! the pose.

use nalgebra::{Matrix3, Vector3};

use crate::landmark::Landmark;

/// Stable landmark indices (out of 468).
const INDICES: [usize; 18] = [
    10, 151, 9, // forehead
    33, 133, 362, 263, // eye corners
    6, 168, 197, 195, 5, // nose bridge / base
    1,   // nose tip
    234, 454, // cheeks
    175, 199, // chin sides
    152, // chin
];

/// Collect the stable landmarks as 3D points.
#[must_use]
pub fn cloud(landmarks: &[Landmark]) -> Vec<Vector3<f32>> {
    INDICES
        .iter()
        .filter_map(|&index| landmarks.get(index))
        .map(|l| Vector3::new(l.x, l.y, l.z))
        .collect()
}

/// Rotation `R` with `current ≈ R * reference` (Kabsch), or `None` when there
/// are too few points or the decomposition fails.
#[must_use]
pub fn rotation(current: &[Vector3<f32>], reference: &[Vector3<f32>]) -> Option<Matrix3<f32>> {
    if current.len() != reference.len() || current.len() < 3 {
        return None;
    }
    let n = current.len() as f32;
    let current_mean = current.iter().sum::<Vector3<f32>>() / n;
    let reference_mean = reference.iter().sum::<Vector3<f32>>() / n;

    let mut covariance = Matrix3::zeros();
    for (c, r) in current.iter().zip(reference) {
        // reference * current^T: with current = R * reference, Kabsch gives
        // R = V * U^T.
        covariance += (r - reference_mean) * (c - current_mean).transpose();
    }

    let svd = covariance.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut correction = Matrix3::identity();
    if (v_t.transpose() * u.transpose()).determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    Some(v_t.transpose() * correction * u.transpose())
}

/// Extract `(yaw, pitch, roll)` in degrees from a rotation matrix built as
/// `Ry(yaw) * Rx(pitch) * Rz(roll)`.
#[must_use]
pub fn angles(r: &Matrix3<f32>) -> (f32, f32, f32) {
    let pitch = (-r[(1, 2)]).asin().to_degrees();
    let yaw = r[(0, 2)].atan2(r[(2, 2)]).to_degrees();
    let roll = r[(1, 0)].atan2(r[(1, 1)]).to_degrees();
    (yaw, pitch, roll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn ry(a: f32) -> Matrix3<f32> {
        Rotation3::from_axis_angle(&Vector3::y_axis(), a).matrix().to_owned()
    }
    fn rx(a: f32) -> Matrix3<f32> {
        Rotation3::from_axis_angle(&Vector3::x_axis(), a).matrix().to_owned()
    }
    fn rz(a: f32) -> Matrix3<f32> {
        Rotation3::from_axis_angle(&Vector3::z_axis(), a).matrix().to_owned()
    }

    #[test]
    fn recovers_known_rotation_and_angles() {
        let reference: Vec<Vector3<f32>> = vec![
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(-0.5, 0.2, 0.8),
        ];

        let (yaw, pitch, roll) = (0.3_f32, -0.2_f32, 0.1_f32);
        let r_true = ry(yaw) * rx(pitch) * rz(roll);
        let current: Vec<Vector3<f32>> = reference.iter().map(|p| r_true * p).collect();

        let r = rotation(&current, &reference).expect("rotation");
        assert!((r - r_true).norm() < 1.0e-3, "recovered rotation differs");

        let (y, p, ro) = angles(&r);
        assert!((y - yaw.to_degrees()).abs() < 0.1, "yaw {y}");
        assert!((p - pitch.to_degrees()).abs() < 0.1, "pitch {p}");
        assert!((ro - roll.to_degrees()).abs() < 0.1, "roll {ro}");
    }
}
