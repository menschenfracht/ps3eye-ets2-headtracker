//! Headtrack — DIY head tracker for ETS2 built on the PS3 Eye (OV534) camera.
//!
//! Modes:
//! - (default) stream the camera and report the measured frame rate;
//! - `--detect` run the face detector headless and save annotated samples;
//! - `--capture` save clean (unannotated) samples for offline analysis;
//! - `--preview` the simple tracker: detection keypoints → pose, live overlay;
//! - `--facemesh` the FaceMesh inspection harness: crop, landmarks, score.

use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

mod async_stream;
mod bayer;
mod camera;
mod face;
mod font;
mod landmark;
mod ov534;
mod pose;
mod scanner;

use camera::{Ps3Eye, Resolution};
use face::FaceDetector;
use landmark::FaceMesh;

/// Stream mode under test (VGA 640x480 at 60 fps: better exposure/quality).
const RESOLUTION: Resolution = Resolution::Vga;
const TARGET_FPS: u16 = 60;
/// Duration of the headless `--detect`/`--capture` benches.
const RUN_FOR: Duration = Duration::from_secs(10);
/// Save one sample frame every this many frames.
const SAMPLE_EVERY: usize = 250;
/// Detector model, relative to the working directory.
const DETECTOR_MODEL: &str = "models/face_detection_yunet.onnx";
/// Landmark model, relative to the working directory.
const LANDMARK_MODEL: &str = "models/face_landmark_192.onnx";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Stream,
    Detect,
    Capture,
    Preview,
    Facemesh,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let mode = parse_mode();

    let mut eye = Ps3Eye::open()?;
    eye.init(RESOLUTION, TARGET_FPS).context("camera init failed")?;
    log::info!("sensor id {:#06x}, target {} fps", eye.sensor_id(), eye.fps());

    eye.start().context("failed to start streaming")?;
    let (width, height) = eye.resolution().context("init missing")?.dims();
    let (width, height) = (
        usize::try_from(width).expect("width fits usize"),
        usize::try_from(height).expect("height fits usize"),
    );

    match mode {
        Mode::Stream => stream_loop(&eye, width, height),
        Mode::Detect => detect_loop(&eye, width, height),
        Mode::Capture => capture_loop(&eye, width, height),
        Mode::Preview => preview_loop(&eye, width, height),
        Mode::Facemesh => facemesh_loop(&eye, width, height),
    }?;

    eye.stop().context("failed to stop streaming")?;
    Ok(())
}

fn parse_mode() -> Mode {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--preview") {
        Mode::Preview
    } else if args.iter().any(|a| a == "--facemesh") {
        Mode::Facemesh
    } else if args.iter().any(|a| a == "--capture") {
        Mode::Capture
    } else if args.iter().any(|a| a == "--detect") {
        Mode::Detect
    } else {
        Mode::Stream
    }
}

/// Default mode: measure and report the camera frame rate.
fn stream_loop(eye: &Ps3Eye, width: usize, height: usize) -> Result<()> {
    let started = Instant::now();
    let mut frames: usize = 0;
    while started.elapsed() < RUN_FOR {
        let raw = eye.read_raw_frame().context("frame read failed")?;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);
        frames += 1;
    }
    let elapsed = started.elapsed();
    let measured = f64::from(u32::try_from(frames).unwrap_or(u32::MAX)) / elapsed.as_secs_f64();
    log::info!(
        "measured {frames} frames in {elapsed:.2?} → {measured:.1} fps (target {})",
        eye.fps()
    );
    Ok(())
}

/// Headless detector bench: annotate and save sample frames.
fn detect_loop(eye: &Ps3Eye, width: usize, height: usize) -> Result<()> {
    let environment = ort::init()
        .with_telemetry(false)
        .build()
        .context("failed to initialise ONNX Runtime")?;
    let mut detector = FaceDetector::load(Path::new(DETECTOR_MODEL), &environment)?;

    let started = Instant::now();
    let mut frames: usize = 0;
    let mut faces_seen: usize = 0;
    while started.elapsed() < RUN_FOR {
        let raw = eye.read_raw_frame().context("frame read failed")?;
        frames += 1;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);

        let faces = detector.detect(&rgb, width, height)?;
        faces_seen += faces.len();

        if frames % SAMPLE_EVERY == 0 {
            for face in &faces {
                draw_rect(&mut rgb, width, height, bx(face.x0), bx(face.y0), bx(face.x1), bx(face.y1), 255, 0, 0);
            }
            let path = format!("docs/m2_{width}x{height}_{frames:05}.ppm");
            save_ppm(&path, &rgb, width, height)?;
            log::info!("sample frame {frames} saved to {path}");
        }
    }

    let elapsed = started.elapsed();
    let measured = f64::from(u32::try_from(frames).unwrap_or(u32::MAX)) / elapsed.as_secs_f64();
    log::info!(
        "measured {frames} frames in {elapsed:.2?} → {measured:.1} fps (target {})",
        eye.fps()
    );
    log::info!("faces detected: {faces_seen} in total over {frames} frames");
    Ok(())
}

/// Data capture: save clean (unannotated) sample frames for offline analysis.
fn capture_loop(eye: &Ps3Eye, width: usize, height: usize) -> Result<()> {
    let started = Instant::now();
    let mut frames: usize = 0;
    while started.elapsed() < RUN_FOR {
        let raw = eye.read_raw_frame().context("frame read failed")?;
        frames += 1;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);
        if frames % SAMPLE_EVERY == 0 {
            let path = format!("docs/cap_{width}x{height}_{frames:05}.ppm");
            save_ppm(&path, &rgb, width, height)?;
            log::info!("clean sample frame {frames} saved to {path}");
        }
    }
    let elapsed = started.elapsed();
    let measured = f64::from(u32::try_from(frames).unwrap_or(u32::MAX)) / elapsed.as_secs_f64();
    log::info!(
        "measured {frames} frames in {elapsed:.2?} → {measured:.1} fps (target {})",
        eye.fps()
    );
    Ok(())
}

/// Simple tracker: detection keypoints → pose, with a live overlay.
///
/// Draws an oval guide to position the head. Press `F` to start a 10-second
/// neutral calibration (hold still); the averaged pose becomes the zero point.
fn preview_loop(eye: &Ps3Eye, width: usize, height: usize) -> Result<()> {
    let environment = ort::init()
        .with_telemetry(false)
        .build()
        .context("failed to initialise ONNX Runtime")?;
    let mut detector = FaceDetector::load(Path::new(DETECTOR_MODEL), &environment)?;

    let mut window = minifb::Window::new(
        "PS3 Eye head tracker (keypoints)",
        width,
        height,
        minifb::WindowOptions {
            scale: minifb::Scale::X2,
            ..minifb::WindowOptions::default()
        },
    )
    .context("failed to open the preview window")?;

    // Oval guide disabled while tuning.
    // let (cx, cy) = (width as f32 * 0.5, height as f32 * 0.5);
    // let (mut rx, mut ry) = (110.0_f32, 140.0_f32);

    let mut buffer = vec![0_u32; width * height];
    let mut frames: usize = 0;
    let mut prev_center: Option<(f32, f32)> = None;
    let mut smoothed_lm: Option<[(f32, f32); 5]> = None;
    let mut smoothed_box: Option<(f32, f32, f32, f32)> = None;
    let mut ref_eye: f32 = 0.0;
    let mut last_pose: Option<pose::Pose> = None;
    let mut neutral: Option<pose::Pose> = None;
    let mut calibrating: Option<Instant> = None;
    let mut calib_samples: Vec<pose::Pose> = Vec::new();
    let mut tuning = pose::Tuning::default();

    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        let raw = eye.read_raw_frame().context("frame read failed")?;
        frames += 1;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);

        // Calibration disabled while tuning; uncomment to re-enable.
        // if window.is_key_down(minifb::Key::F) && calibrating.is_none() {
        //     calibrating = Some(Instant::now());
        //     calib_samples.clear();
        //     log::info!("calibration started — hold still for 10 s");
        // }

        // Single-press recenter: capture the current pose as neutral.
        if window.is_key_pressed(minifb::Key::R, minifb::KeyRepeat::No) {
            if let Some(p) = last_pose {
                neutral = Some(p);
                log::info!(
                    "recentered: neutral yaw={:+.1}° pitch={:+.1}° roll={:+.1}°",
                    p.yaw,
                    p.pitch,
                    p.roll
                );
            }
        }

        // Oval resize disabled while tuning.
        // if window.is_key_pressed(minifb::Key::Equal, minifb::KeyRepeat::No)
        //     || window.is_key_pressed(minifb::Key::NumPadPlus, minifb::KeyRepeat::No)
        // {
        //     rx *= 1.1;
        //     ry *= 1.1;
        //     log::info!("oval enlarged to {rx:.0} x {ry:.0} px");
        // }
        // if window.is_key_pressed(minifb::Key::Minus, minifb::KeyRepeat::No)
        //     || window.is_key_pressed(minifb::Key::NumPadMinus, minifb::KeyRepeat::No)
        // {
        //     rx = (rx / 1.1).max(30.0_f32);
        //     ry = (ry / 1.1).max(40.0_f32);
        //     log::info!("oval shrunk to {rx:.0} x {ry:.0} px");
        // }

        // Oval guide disabled while tuning.
        // let (or, og, ob) = if calibrating.is_some() {
        //     (255_u8, 165, 0)
        // } else {
        //     (0_u8, 255, 0)
        // };
        // draw_oval(&mut rgb, width, height, cx, cy, rx, ry, or, og, ob);

        // Tuning keys: [ ] yaw gain, ; ' pitch gain, , . deadzone,
        // n m smoothing, i/o/p invert yaw/pitch/roll.
        if window.is_key_pressed(minifb::Key::RightBracket, minifb::KeyRepeat::No) {
            tuning.yaw_gain = (tuning.yaw_gain + 0.1).min(6.0);
            log::info!("yaw_gain {:.1}", tuning.yaw_gain);
        }
        if window.is_key_pressed(minifb::Key::LeftBracket, minifb::KeyRepeat::No) {
            tuning.yaw_gain = (tuning.yaw_gain - 0.1).max(0.5);
            log::info!("yaw_gain {:.1}", tuning.yaw_gain);
        }
        if window.is_key_pressed(minifb::Key::Apostrophe, minifb::KeyRepeat::No) {
            tuning.pitch_gain = (tuning.pitch_gain + 0.1).min(6.0);
            log::info!("pitch_gain {:.1}", tuning.pitch_gain);
        }
        if window.is_key_pressed(minifb::Key::Semicolon, minifb::KeyRepeat::No) {
            tuning.pitch_gain = (tuning.pitch_gain - 0.1).max(0.5);
            log::info!("pitch_gain {:.1}", tuning.pitch_gain);
        }
        if window.is_key_pressed(minifb::Key::Period, minifb::KeyRepeat::No) {
            tuning.deadzone = (tuning.deadzone + 0.5).min(15.0);
            log::info!("deadzone {:.1}°", tuning.deadzone);
        }
        if window.is_key_pressed(minifb::Key::Comma, minifb::KeyRepeat::No) {
            tuning.deadzone = (tuning.deadzone - 0.5).max(0.0);
            log::info!("deadzone {:.1}°", tuning.deadzone);
        }
        if window.is_key_pressed(minifb::Key::M, minifb::KeyRepeat::No) {
            tuning.smoothing = (tuning.smoothing + 0.05).min(1.0);
            log::info!("smoothing {:.2}", tuning.smoothing);
        }
        if window.is_key_pressed(minifb::Key::N, minifb::KeyRepeat::No) {
            tuning.smoothing = (tuning.smoothing - 0.05).max(0.05);
            log::info!("smoothing {:.2}", tuning.smoothing);
        }
        if window.is_key_pressed(minifb::Key::I, minifb::KeyRepeat::No) {
            tuning.invert_yaw = !tuning.invert_yaw;
            log::info!("invert_yaw {}", tuning.invert_yaw);
        }
        if window.is_key_pressed(minifb::Key::O, minifb::KeyRepeat::No) {
            tuning.invert_pitch = !tuning.invert_pitch;
            log::info!("invert_pitch {}", tuning.invert_pitch);
        }
        if window.is_key_pressed(minifb::Key::P, minifb::KeyRepeat::No) {
            tuning.invert_roll = !tuning.invert_roll;
            log::info!("invert_roll {}", tuning.invert_roll);
        }

        let faces = detector.detect(&rgb, width, height)?;
        if let Some(face) = select_face(&faces, prev_center) {
            let eye = face.eye_distance();
            ref_eye = (ref_eye * 0.999).max(eye);
            // Freeze when the detected "face" is much smaller than the tracked
            // one: at extreme angles YuNet sees only a sub-region.
            if eye >= 0.6 * ref_eye {
                let landmarks = match smoothed_lm {
                    Some(prev) => blend_landmarks(prev, face.landmarks, tuning.smoothing),
                    None => face.landmarks,
                };
                smoothed_lm = Some(landmarks);

                let raw_box = (face.x0, face.y0, face.x1, face.y1);
                let box_ = match smoothed_box {
                    Some(prev) => blend_box(prev, raw_box, tuning.smoothing),
                    None => raw_box,
                };
                smoothed_box = Some(box_);
                prev_center = Some(face.center());
            }
        }

        // Draw the overlay from the (possibly frozen) smoothed values.
        if let (Some(landmarks), Some(box_)) = (smoothed_lm, smoothed_box) {
            draw_rect(&mut rgb, width, height, bx(box_.0), bx(box_.1), bx(box_.2), bx(box_.3), 255, 0, 0);

            let [eye_left, eye_right, nose, _mouth_r, _mouth_l] = landmarks;
            let eye_dist = (eye_right.0 - eye_left.0).hypot(eye_right.1 - eye_left.1);

            // Eye line (roll) in yellow, nose direction (yaw/pitch) in cyan.
            draw_line(&mut rgb, width, height, eye_left, eye_right, 255, 255, 0);
            let eye_mid = (
                (eye_left.0 + eye_right.0) * 0.5,
                (eye_left.1 + eye_right.1) * 0.5,
            );
            draw_line(&mut rgb, width, height, eye_mid, nose, 0, 255, 255);

            for point in landmarks {
                draw_dot(&mut rgb, width, height, point.0, point.1, 0, 255, 0);
            }

            let pose = pose::estimate(&[eye_left, eye_right, nose]);
            last_pose = Some(pose);

            if let Some(start) = calibrating {
                calib_samples.push(pose);
                if start.elapsed() >= Duration::from_secs(10) {
                    neutral = Some(mean_pose(&calib_samples));
                    calibrating = None;
                    if let Some(n) = neutral {
                        log::info!(
                            "calibration done: neutral yaw={:+.1}° pitch={:+.1}° roll={:+.1}° ({} samples)",
                            n.yaw,
                            n.pitch,
                            n.roll,
                            calib_samples.len()
                        );
                    }
                }
            }

            let shown = tuning.apply(pose, neutral);
            let state = if calibrating.is_some() {
                "CALIBRATING"
            } else if neutral.is_some() {
                "TRACKING"
            } else {
                "UNCALIBRATED"
            };
            draw_hud(&mut rgb, width, height, state, eye_dist, shown, &tuning);
            if frames % TARGET_FPS as usize == 0 {
                log::info!(
                    "{state:12} eye_dist={eye_dist:4.1}px  yaw {:>6.1}  pitch {:>6.1}  roll {:>6.1}  [gain {:.1}/{:.1} dz {:.1} sm {:.2}]",
                    shown.yaw,
                    shown.pitch,
                    shown.roll,
                    tuning.yaw_gain,
                    tuning.pitch_gain,
                    tuning.deadzone,
                    tuning.smoothing
                );
            }
        }

        rgb_to_u32(&rgb, width, height, &mut buffer);
        window
            .update_with_buffer(&buffer, width, height)
            .context("failed to update the preview window")?;
    }
    Ok(())
}

/// Pick the largest face (the user's), using the previous centre only as a
/// tie-break. Sub-region false detections are always smaller, so this rejects
/// them outright.
fn select_face<'a>(faces: &'a [face::Face], prev_center: Option<(f32, f32)>) -> Option<&'a face::Face> {
    let max_area = faces.iter().map(box_area).fold(0.0_f32, f32::max);
    if max_area <= 0.0 {
        return None;
    }
    let candidates: Vec<&face::Face> = faces
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
fn box_area(f: &face::Face) -> f32 {
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

/// Mean of a set of pose samples.
fn mean_pose(samples: &[pose::Pose]) -> pose::Pose {
    if samples.is_empty() {
        return pose::Pose::default();
    }
    let n = samples.len() as f32;
    let sum = samples
        .iter()
        .fold(pose::Pose::default(), |acc, p| pose::Pose {
            yaw: acc.yaw + p.yaw,
            pitch: acc.pitch + p.pitch,
            roll: acc.roll + p.roll,
        });
    pose::Pose {
        yaw: sum.yaw / n,
        pitch: sum.pitch / n,
        roll: sum.roll / n,
    }
}

/// Draw the outline of an ellipse (the head-position guide).
#[allow(dead_code, reason = "oval guide is disabled while tuning")]
#[expect(
    clippy::indexing_slicing,
    reason = "pixel offsets are within the validated frame dimensions"
)]
fn draw_oval(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
    r: u8,
    g: u8,
    b: u8,
) {
    let rx2 = rx * rx;
    let ry2 = ry * ry;
    for y in 0..height {
        let dy = y as f32 - cy;
        for x in 0..width {
            let dx = x as f32 - cx;
            let v = dx * dx / rx2 + dy * dy / ry2;
            if (v - 1.0).abs() < 0.012 {
                put_px(rgb, width, height, x, y, r, g, b);
            }
        }
    }
}

/// FaceMesh inspection harness: show the crop region and landmarks so the
/// face size and alignment can be checked by eye.
fn facemesh_loop(eye: &Ps3Eye, width: usize, height: usize) -> Result<()> {
    let environment = ort::init()
        .with_telemetry(false)
        .build()
        .context("failed to initialise ONNX Runtime")?;
    let mut detector = FaceDetector::load(Path::new(DETECTOR_MODEL), &environment)?;
    let mut mesh = FaceMesh::load(Path::new(LANDMARK_MODEL), &environment)?;

    let mut window = minifb::Window::new(
        "PS3 Eye FaceMesh harness",
        width,
        height,
        minifb::WindowOptions {
            scale: minifb::Scale::X2,
            ..minifb::WindowOptions::default()
        },
    )
    .context("failed to open the preview window")?;

    let mut buffer = vec![0_u32; width * height];
    let mut frames: usize = 0;
    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        let raw = eye.read_raw_frame().context("frame read failed")?;
        frames += 1;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);

        let faces = detector.detect(&rgb, width, height)?;
        if let Some(face) = faces.first() {
            draw_rect(&mut rgb, width, height, bx(face.x0), bx(face.y0), bx(face.x1), bx(face.y1), 255, 0, 0);

            let left_eye = face.landmarks[1];
            let right_eye = face.landmarks[0];
            let eye_dist = (right_eye.0 - left_eye.0).hypot(right_eye.1 - left_eye.1);
            let rect = face::face_rect(face);

            // Crop region (blue square) and its centre.
            let (cx_r, cy_r, half) = (rect.cx, rect.cy, rect.half);
            draw_rect(&mut rgb, width, height, bx(cx_r - half), bx(cy_r - half), bx(cx_r + half), bx(cy_r + half), 0, 0, 255);

            let (landmarks, score) = mesh.run(&rgb, width, height, &rect)?;
            for lm in &landmarks {
                let (sx, sy) = mesh.to_source(lm.x, lm.y, &rect);
                draw_dot(&mut rgb, width, height, sx, sy, 0, 255, 0);
            }
            if frames % TARGET_FPS as usize == 0 {
                log::info!(
                    "eye_dist={eye_dist:4.1}px  crop_side={:.0}px  score={score:6.1}",
                    half * 2.0
                );
            }
        }

        rgb_to_u32(&rgb, width, height, &mut buffer);
        window
            .update_with_buffer(&buffer, width, height)
            .context("failed to update the preview window")?;
    }
    Ok(())
}

/// Clamp a float coordinate to a non-negative pixel index.
fn bx(v: f32) -> usize {
    v.max(0.0) as usize
}

/// Put a single pixel with the given colour (clipped to the frame).
#[expect(
    clippy::indexing_slicing,
    reason = "pixel offset is validated against the frame dimensions"
)]
fn put_px(rgb: &mut [u8], width: usize, height: usize, x: usize, y: usize, r: u8, g: u8, b: u8) {
    if x < width && y < height {
        let i = (y * width + x) * 3;
        if let Some([pr, pg, pb]) = rgb.get_mut(i..i + 3) {
            *pr = r;
            *pg = g;
            *pb = b;
        }
    }
}

/// Draw a 2-pixel rectangle onto an interleaved RGB8 buffer.
fn draw_rect(rgb: &mut [u8], width: usize, height: usize, x0: usize, y0: usize, x1: usize, y1: usize, r: u8, g: u8, b: u8) {
    let x1 = x1.min(width.saturating_sub(1));
    let y1 = y1.min(height.saturating_sub(1));
    for x in x0..=x1 {
        put_px(rgb, width, height, x, y0, r, g, b);
        put_px(rgb, width, height, x, y1, r, g, b);
        put_px(rgb, width, height, x, y0 + 1, r, g, b);
        put_px(rgb, width, height, x, y1.saturating_sub(1), r, g, b);
    }
    for y in y0..=y1 {
        put_px(rgb, width, height, x0, y, r, g, b);
        put_px(rgb, width, height, x1, y, r, g, b);
        put_px(rgb, width, height, x0 + 1, y, r, g, b);
        put_px(rgb, width, height, x1.saturating_sub(1), y, r, g, b);
    }
}

/// Draw a line with Bresenham's algorithm.
fn draw_line(rgb: &mut [u8], width: usize, height: usize, a: (f32, f32), b: (f32, f32), r: u8, g: u8, bl: u8) {
    let (mut x0, mut y0) = (a.0.round() as i64, a.1.round() as i64);
    let (x1, y1) = (b.0.round() as i64, b.1.round() as i64);
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 {
            put_px(rgb, width, height, x0 as usize, y0 as usize, r, g, bl);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a small 3x3 dot at `(x, y)`.
fn draw_dot(rgb: &mut [u8], width: usize, height: usize, x: f32, y: f32, r: u8, g: u8, b: u8) {
    let x = x.round() as usize;
    let y = y.round() as usize;
    for dy in 0_usize..3 {
        for dx in 0_usize..3 {
            put_px(rgb, width, height, x + dx, y + dy, r, g, b);
        }
    }
}

/// Draw a string with the 5x7 font, `scale` screen pixels per font pixel.
#[expect(
    clippy::indexing_slicing,
    reason = "glyph index is bounded by the ASCII 32..=126 check"
)]
fn draw_text(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    scale: usize,
    r: u8,
    g: u8,
    b: u8,
) {
    let mut cx = x;
    for ch in text.bytes() {
        if (32..=126).contains(&ch) {
            let glyph = font::FONT5X7[usize::from(ch - 32)];
            for (col, &bits) in glyph.iter().enumerate() {
                for row in 0..7 {
                    if bits & (1 << row) != 0 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                put_px(
                                    rgb,
                                    width,
                                    height,
                                    cx + col * scale + dx,
                                    y + row * scale + dy,
                                    r,
                                    g,
                                    b,
                                );
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

/// Draw a string with a 1px black shadow so it reads over video.
fn draw_text_shadow(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    scale: usize,
    r: u8,
    g: u8,
    b: u8,
) {
    draw_text(rgb, width, height, x + 1, y + 1, text, scale, 0, 0, 0);
    draw_text(rgb, width, height, x, y, text, scale, r, g, b);
}

/// Draw the tuning HUD: state, pose, settings and key help.
fn draw_hud(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    state: &str,
    eye_dist: f32,
    shown: pose::Pose,
    tuning: &pose::Tuning,
) {
    let yes = |v: bool| if v { "Y" } else { "N" };
    let line1 = format!(
        "{state}  YAW {:+5.1}  PITCH {:+5.1}  ROLL {:+5.1}",
        shown.yaw, shown.pitch, shown.roll
    );
    draw_text_shadow(rgb, width, height, 6, 6, &line1, 2, 255, 255, 255);

    let line2 = format!(
        "EYES {eye_dist:4.0}px  GAIN {:.1}/{:.1}  DZ {:.1}  SM {:.2}  INV {}/{}/{}",
        tuning.yaw_gain,
        tuning.pitch_gain,
        tuning.deadzone,
        tuning.smoothing,
        yes(tuning.invert_yaw),
        yes(tuning.invert_pitch),
        yes(tuning.invert_roll)
    );
    draw_text_shadow(rgb, width, height, 6, 32, &line2, 1, 0, 255, 200);

    draw_text_shadow(
        rgb,
        width,
        height,
        6,
        height - 36,
        "R RECENTER   [ ] YAW GAIN   ; ' PITCH GAIN   , . DEADZONE",
        1,
        220,
        220,
        220,
    );
    draw_text_shadow(
        rgb,
        width,
        height,
        6,
        height - 22,
        "N M SMOOTH   I O P INVERT   ESC QUIT",
        1,
        220,
        220,
        220,
    );
}

/// Convert an interleaved RGB8 frame to a `0x00RRGGBB` buffer for minifb.
#[expect(
    clippy::indexing_slicing,
    reason = "pixel offsets are within the validated frame and output buffers"
)]
fn rgb_to_u32(rgb: &[u8], width: usize, height: usize, out: &mut [u32]) {
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            let r = u32::from(rgb[i]);
            let g = u32::from(rgb[i + 1]);
            let b = u32::from(rgb[i + 2]);
            out[y * width + x] = (r << 16) | (g << 8) | b;
        }
    }
}

/// Write a binary PPM (P6) — no image library needed for diagnostics.
fn save_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> Result<()> {
    let mut file = std::fs::File::create(path).with_context(|| format!("create {path}"))?;
    writeln!(file, "P6\n{width} {height}\n255")?;
    file.write_all(rgb).with_context(|| format!("write {path}"))?;
    Ok(())
}
