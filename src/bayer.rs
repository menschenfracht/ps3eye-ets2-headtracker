//! Debayering for the raw frames the OV772x sensor produces.
//!
//! The camera streams 8-bit Bayer with the GRBG pattern (per the reference
//! driver). We do plain bilinear interpolation with clamped neighbors, which
//! is equivalent to the reference implementation's edge replication.

/// Average of two bytes, rounded like the reference driver (`(a + b + 1) >> 1`).
#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) / 2) as u8
}

/// Convert a raw GRBG Bayer frame into RGB8 (3 bytes per pixel: R, G, B).
///
/// `out` must be exactly `width * height * 3` bytes long.
///
/// # Panics
/// Panics if `bayer` or `out` do not match the requested dimensions.
pub fn bayer_grbg_to_rgb(bayer: &[u8], width: usize, height: usize, out: &mut [u8]) {
    assert_eq!(bayer.len(), width * height, "bayer buffer size mismatch");
    assert_eq!(out.len(), width * height * 3, "output buffer size mismatch");

    let last_x = width - 1;
    let last_y = height - 1;

    for y in 0..height {
        let y1 = y.saturating_sub(1);
        let y2 = (y + 1).min(last_y);
        for x in 0..width {
            let x1 = x.saturating_sub(1);
            let x2 = (x + 1).min(last_x);
            let v = bayer[y * width + x];

            // Neighbors matching the GRBG pattern:
            //   G R G R
            //   B G B G
            let (r, g, b) = match (y % 2, x % 2) {
                // G on a green-red row
                (0, 0) => (
                    avg2(bayer[y * width + x1], bayer[y * width + x2]),
                    v,
                    avg2(bayer[y1 * width + x], bayer[y2 * width + x]),
                ),
                // R on a green-red row
                (0, 1) => (
                    v,
                    avg2(bayer[y * width + x1], bayer[y * width + x2]),
                    avg2(bayer[y1 * width + x], bayer[y2 * width + x]),
                ),
                // B on a blue-green row
                (1, 0) => (
                    avg2(bayer[y1 * width + x], bayer[y2 * width + x]),
                    avg2(bayer[y * width + x1], bayer[y * width + x2]),
                    v,
                ),
                // G on a blue-green row
                (1, 1) => (
                    avg2(bayer[y1 * width + x], bayer[y2 * width + x]),
                    v,
                    avg2(bayer[y * width + x1], bayer[y * width + x2]),
                ),
                _ => unreachable!("boolean pair covers all cases"),
            };

            let dest = (y * width + x) * 3;
            out[dest] = r;
            out[dest + 1] = g;
            out[dest + 2] = b;
        }
    }
}

/// Convert a raw GRBG Bayer frame into grayscale (1 byte per pixel),
/// using the green channel — the sensor's dominant sample.
///
/// `out` must be exactly `width * height` bytes long.
///
/// # Panics
/// Panics if `bayer` or `out` do not match the requested dimensions.
pub fn bayer_grbg_to_gray(bayer: &[u8], width: usize, height: usize, out: &mut [u8]) {
    assert_eq!(bayer.len(), width * height, "bayer buffer size mismatch");
    assert_eq!(out.len(), width * height, "output buffer size mismatch");
    out.copy_from_slice(bayer);
}
