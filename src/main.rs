//! Headtrack — DIY head tracker for ETS2 built on the PS3 Eye (OV534) camera.
//!
//! Milestone M1.1: bring the camera up with our own OV534 port, stream for a
//! fixed period, report the measured frame rate, and save sample frames.

use std::io::Write as _;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

mod async_stream;
mod bayer;
mod camera;
mod ov534;
mod scanner;

use camera::{Ps3Eye, Resolution};

/// Stream mode under test (QVGA 125 fps: the nearest table row to 120).
const RESOLUTION: Resolution = Resolution::Qvga;
const TARGET_FPS: u16 = 125;

/// How long the stability run lasts.
const RUN_FOR: Duration = Duration::from_secs(60);
/// Save one sample frame every this many frames.
const SAMPLE_EVERY: usize = 600;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

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
    while started.elapsed() < RUN_FOR {
        let raw = eye.read_raw_frame().context("frame read failed")?;
        frames += 1;
        if frames % SAMPLE_EVERY == 0 {
            let mut rgb = vec![0_u8; width * height * 3];
            bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);
            let path = format!("D:\\_coding\\Headtrack\\docs\\async_{width}x{height}_{frames:05}.ppm");
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

    eye.stop().context("failed to stop streaming")?;
    Ok(())
}

/// Write a binary PPM (P6) — no image library needed for diagnostics.
fn save_ppm(path: &str, rgb: &[u8], width: usize, height: usize) -> Result<()> {
    let mut file = std::fs::File::create(path).with_context(|| format!("create {path}"))?;
    writeln!(file, "P6\n{width} {height}\n255")?;
    file.write_all(rgb).with_context(|| format!("write {path}"))?;
    Ok(())
}
