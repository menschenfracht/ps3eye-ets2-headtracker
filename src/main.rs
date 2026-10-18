//! Headtrack — DIY head tracker for ETS2 built on the PS3 Eye (OV534) camera.
//!
//! Stream the camera, optionally run the face detector on every frame, and
//! save sample frames. `--detect` enables the detector.

use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

mod async_stream;
mod bayer;
mod camera;
mod face;
mod ov534;
mod scanner;

use camera::{Ps3Eye, Resolution};
use face::{FaceDetector, Letterbox};

/// Stream mode under test (QVGA 125 fps: the nearest table row to 120).
const RESOLUTION: Resolution = Resolution::Qvga;
const TARGET_FPS: u16 = 125;
/// How long a run lasts.
const RUN_FOR: Duration = Duration::from_secs(10);
/// Save one sample frame every this many frames.
const SAMPLE_EVERY: usize = 250;
/// Detector model, relative to the working directory.
const DETECTOR_MODEL: &str = "models/face_detection_short_range.onnx";

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let detect = std::env::args().any(|arg| arg == "--detect");

    let mut detector = if detect {
        let environment = ort::init()
            .with_telemetry(false)
            .build()
            .context("failed to initialise ONNX Runtime")?;
        Some(FaceDetector::load(Path::new(DETECTOR_MODEL), &environment)?)
    } else {
        None
    };

    let mut eye = Ps3Eye::open()?;
    eye.init(RESOLUTION, TARGET_FPS).context("camera init failed")?;
    log::info!("sensor id {:#06x}, target {} fps", eye.sensor_id(), eye.fps());

    eye.start().context("failed to start streaming")?;
    let (width, height) = eye.resolution().context("init missing")?.dims();
    let (width, height) = (
        usize::try_from(width).expect("width fits usize"),
        usize::try_from(height).expect("height fits usize"),
    );

    let started = Instant::now();
    let mut frames: usize = 0;
    let mut faces_seen: usize = 0;
    while started.elapsed() < RUN_FOR {
        let raw = eye.read_raw_frame().context("frame read failed")?;
        frames += 1;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);

        let mut boxes = Vec::new();
        if let Some(detector) = detector.as_mut() {
            let (faces, letterbox) = detector.detect(&rgb, width, height)?;
            faces_seen += faces.len();
            if let Some(best) = faces.first() {
                log::debug!("frame {frames}: {} face(s), best score {:.2}", faces.len(), best.score);
            }
            boxes = faces
                .iter()
                .map(|face| to_source_box(face, &letterbox))
                .collect();
        }

        if frames % SAMPLE_EVERY == 0 {
            for &(x0, y0, x1, y1) in &boxes {
                draw_rect(&mut rgb, width, height, x0, y0, x1, y1);
            }
            let path = format!("D:\\_coding\\Headtrack\\docs\\m2_{width}x{height}_{frames:05}.ppm");
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
    if detect {
        log::info!("faces detected: {faces_seen} in total over {frames} frames");
    }

    eye.stop().context("failed to stop streaming")?;
    Ok(())
}

/// Convert a detector box (letterbox-normalised) to source pixel corners.
fn to_source_box(face: &face::Face, letterbox: &Letterbox) -> (usize, usize, usize, usize) {
    let (x0, y0) = letterbox.to_source(face.x0, face.y0);
    let (x1, y1) = letterbox.to_source(face.x1, face.y1);
    let clamp = |v: f32| v.max(0.0) as usize;
    (clamp(x0), clamp(y0), clamp(x1), clamp(y1))
}

/// Draw a 2-pixel rectangle onto an interleaved RGB8 buffer.
fn draw_rect(rgb: &mut [u8], width: usize, height: usize, x0: usize, y0: usize, x1: usize, y1: usize) {
    let x1 = x1.min(width.saturating_sub(1));
    let y1 = y1.min(height.saturating_sub(1));
    let mut put = |x: usize, y: usize| {
        if x < width && y < height {
            let i = (y * width + x) * 3;
            if let Some([r, g, b]) = rgb.get_mut(i..i + 3) {
                *r = 255;
                *g = 0;
                *b = 0;
            }
        }
    };
    for x in x0..=x1 {
        put(x, y0);
        put(x, y1);
        put(x, y0 + 1);
        put(x, y1.saturating_sub(1));
    }
    for y in y0..=y1 {
        put(x0, y);
        put(x1, y);
        put(x0 + 1, y);
        put(x1.saturating_sub(1), y);
    }
}

/// Write a binary PPM (P6) — no image library needed for diagnostics.
fn save_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> Result<()> {
    let mut file = std::fs::File::create(path).with_context(|| format!("create {path}"))?;
    writeln!(file, "P6\n{width} {height}\n255")?;
    file.write_all(rgb).with_context(|| format!("write {path}"))?;
    Ok(())
}
