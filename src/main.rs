//! Headtrack — DIY head tracker for ETS2 built on the PS3 Eye (OV534) camera.
//!
//! Modes:
//! - (default) stream the camera and report the measured frame rate;
//! - `--detect` run the face detector headless and save annotated samples;
//! - `--capture` save clean (unannotated) samples for offline analysis;
//! - `--preview` the tracker with a live overlay and tuning keys;
//! - `--record` guided capture of a fixed head-move sequence to `docs/pose.csv`;
//! - `--track` the tracker headless: pose → opentrack over UDP, no window;
//! - `--facemesh` the FaceMesh inspection harness: crop, landmarks, score.

use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

mod async_stream;
mod bayer;
mod camera;
mod face;
mod filter;
mod font;
mod hotkey;
mod landmark;
mod output;
mod ov534;
mod pose;
mod pose3d;
mod scanner;
mod tracker;

use camera::{Ps3Eye, Resolution};
use face::FaceDetector;
use landmark::FaceMesh;
use tracker::Tracker;

/// Stream mode under test (VGA 640x480 at 60 fps: better exposure/quality).
const RESOLUTION: Resolution = Resolution::Vga;
const TARGET_FPS: u16 = 30;
/// Duration of the headless `--detect`/`--capture` benches.
const RUN_FOR: Duration = Duration::from_secs(10);
/// Save one sample frame every this many frames.
const SAMPLE_EVERY: usize = 250;
/// Detector model, relative to the working directory.
const DETECTOR_MODEL: &str = "models/face_detection_yunet.onnx";
/// Landing model, relative to the working directory.
const LANDMARK_MODEL: &str = "models/face_landmark_192.onnx";
/// Creating this file stops the headless `--track` loop cleanly.
const STOP_FILE: &str = "target/stop";

/// Guided head poses for `--record`: a neutral, the six directions of the
/// on-screen cross, then neutral again as a return-to-centre check.
const RECORD_STAGES: [&str; 8] = [
    "straight",
    "up",
    "down",
    "left",
    "right",
    "tilt left",
    "tilt right",
    "straight",
];
/// Seconds spent in each guided pose.
const STAGE_SECS: f32 = 4.0;
/// Countdown after Space before the first stage starts.
const COUNTDOWN_SECS: f32 = 3.0;
/// Guided capture output (one row per frame).
const RECORD_FILE: &str = "docs/pose.csv";
/// Mirror the preview image horizontally (camera view → mirror view), so head
/// motion on screen matches what the user feels. Flip to `false` to see the
/// camera's own view. The pose itself is always computed from the unmirrored
/// frame.
const MIRROR_PREVIEW: bool = true;

/// Control-panel size.
const CONTROL_W: usize = 260;
const CONTROL_H: usize = 92;
/// Button layout.
const BTN_W: usize = 116;
const BTN_H: usize = 44;
const BTN_Y: usize = 34;
const RECENTER_X: usize = 12;
const QUIT_X: usize = 132;

/// Every cue is one of these tones, shared by every mode: startup, recenter
/// and stage changes differ in pitch, but the same event always sounds the
/// same no matter which mode triggered it.
const BEEP_START: (u32, u32) = (1000, 250);
const BEEP_RECENTER: (u32, u32) = (1500, 100);
const BEEP_STAGE: (u32, u32) = (2000, 60);

/// Short audible cue (Windows `Beep`, falling back to the system sound).
#[cfg(windows)]
fn beep((frequency, millis): (u32, u32)) {
    // SAFETY: `Beep`/`MessageBeep` have no memory-safety requirements.
    let ok = unsafe { Beep(frequency, millis) };
    if ok == 0 {
        unsafe {
            MessageBeep(0);
        }
    }
}

/// No-op on non-Windows targets.
#[cfg(not(windows))]
fn beep(_tone: (u32, u32)) {}

#[cfg(windows)]
unsafe extern "system" {
    fn Beep(frequency: u32, duration: u32) -> i32;
    fn MessageBeep(beep_type: u32) -> i32;
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Stream,
    Detect,
    Capture,
    Preview,
    Record,
    Track,
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
        Mode::Preview => preview_loop(&eye, width, height, false),
        Mode::Record => preview_loop(&eye, width, height, true),
        Mode::Track => track_loop(&eye, width, height),
        Mode::Facemesh => facemesh_loop(&eye, width, height),
    }?;

    eye.stop().context("failed to stop streaming")?;
    Ok(())
}

fn parse_mode() -> Mode {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--preview") {
        Mode::Preview
    } else if args.iter().any(|a| a == "--record") {
        Mode::Record
    } else if args.iter().any(|a| a == "--track") {
        Mode::Track
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

/// Whether the given command-line switch was passed.
fn has_flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Once-a-second frame rate, dropped-frame and timing report.
struct LoopStats {
    last: Instant,
    frames: u32,
    last_dropped: usize,
    read: Duration,
    work: Duration,
}

impl LoopStats {
    fn new() -> Self {
        Self {
            last: Instant::now(),
            frames: 0,
            last_dropped: 0,
            read: Duration::ZERO,
            work: Duration::ZERO,
        }
    }

    /// Record one loop iteration; logs when a second has passed.
    fn sample(&mut self, eye: &Ps3Eye, read: Duration, work: Duration) {
        self.frames += 1;
        self.read += read;
        self.work += work;
        let elapsed = self.last.elapsed();
        if elapsed < Duration::from_secs(1) {
            return;
        }
        let n = f64::from(self.frames);
        let dropped = eye.skipped_frames();
        log::info!(
            "{:.1} fps  dropped {} ({dropped} total)  read {:.1}ms  work {:.1}ms",
            n / elapsed.as_secs_f64(),
            dropped.saturating_sub(self.last_dropped),
            self.read.as_secs_f64() * 1000.0 / n,
            self.work.as_secs_f64() * 1000.0 / n,
        );
        self.last = Instant::now();
        self.frames = 0;
        self.last_dropped = dropped;
        self.read = Duration::ZERO;
        self.work = Duration::ZERO;
    }
}

/// One arm of the guided-capture cross.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
    RollLeft,
    RollRight,
}

/// Which cross arm a `RECORD_STAGES` entry asks for.
fn stage_dir(stage: usize) -> Option<Dir> {
    match RECORD_STAGES.get(stage).copied() {
        Some("up") => Some(Dir::Up),
        Some("down") => Some(Dir::Down),
        Some("left") => Some(Dir::Left),
        Some("right") => Some(Dir::Right),
        Some("tilt left") => Some(Dir::RollLeft),
        Some("tilt right") => Some(Dir::RollRight),
        _ => None,
    }
}

/// Per-frame CSV log for the guided capture.
struct Recorder {
    out: std::io::BufWriter<std::fs::File>,
}

impl Recorder {
    fn create(path: &str) -> Result<Self> {
        let file = std::fs::File::create(path).with_context(|| format!("failed to create {path}"))?;
        let mut out = std::io::BufWriter::new(file);
        writeln!(
            out,
            "t_s,stage,move,eye_l_x,eye_l_y,eye_r_x,eye_r_y,nose_x,nose_y,\
             mouth_r_x,mouth_r_y,mouth_l_x,mouth_l_y,\
             box_x0,box_y0,box_x1,box_y1,eye_dist,\
             yaw_raw,pitch_raw,roll_raw,yaw,pitch,roll"
        )?;
        Ok(Self { out })
    }

    fn row(&mut self, t: f32, stage: usize, frame: &tracker::Frame) -> Result<()> {
        let [el, er, nose, mr, ml] = frame.landmarks;
        writeln!(
            self.out,
            "{t:.3},{stage},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},\
             {:.2},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            RECORD_STAGES.get(stage).copied().unwrap_or("?"),
            el.0,
            el.1,
            er.0,
            er.1,
            nose.0,
            nose.1,
            mr.0,
            mr.1,
            ml.0,
            ml.1,
            frame.box_.0,
            frame.box_.1,
            frame.box_.2,
            frame.box_.3,
            frame.eye_dist,
            frame.raw_pose.yaw,
            frame.raw_pose.pitch,
            frame.raw_pose.roll,
            frame.pose.yaw,
            frame.pose.pitch,
            frame.pose.roll
        )?;
        Ok(())
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
fn preview_loop(eye: &Ps3Eye, width: usize, height: usize, record: bool) -> Result<()> {
    let environment = ort::init()
        .with_telemetry(false)
        .build()
        .context("failed to initialise ONNX Runtime")?;
    let mut tracker = Tracker::load(
        Path::new(DETECTOR_MODEL),
        Path::new(LANDMARK_MODEL),
        &environment,
        output::DEFAULT_PORT,
    )?;
    tracker.use_3d |= has_flag("--3d");
    log::info!("sending pose to opentrack at 127.0.0.1:{}", output::DEFAULT_PORT);
    let mut recorder = if record {
        log::info!("guided capture → {RECORD_FILE}; follow the highlighted arrow");
        log::info!(
            "sequence: {}",
            RECORD_STAGES
                .iter()
                .enumerate()
                .map(|(i, name)| format!("{}. {name}", i + 1))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Some(Recorder::create(RECORD_FILE)?)
    } else {
        None
    };
    let mut armed: Option<Instant> = None;
    let mut last_stage: Option<usize> = None;

    let mut window = minifb::Window::new(
        "PS3 Eye head tracker",
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
    let mut stats = LoopStats::new();

    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        let read_start = Instant::now();
        let raw = eye.read_raw_frame().context("frame read failed")?;
        let read = read_start.elapsed();
        let work_start = Instant::now();
        frames += 1;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);

        // Recenter: capture the current pose as the neutral reference.
        if window.is_key_pressed(minifb::Key::R, minifb::KeyRepeat::No) {
            tracker.recenter();
            log::info!("recentered");
            beep(BEEP_RECENTER);
        }

        // Tuning keys: [ ] yaw gain, ; ' pitch gain, , . deadzone,
        // n m smoothing, i/o/p invert yaw/pitch/roll.
        if window.is_key_pressed(minifb::Key::RightBracket, minifb::KeyRepeat::No) {
            tracker.tuning.yaw_gain = (tracker.tuning.yaw_gain + 0.1).min(6.0);
            log::info!("yaw_gain {:.1}", tracker.tuning.yaw_gain);
        }
        if window.is_key_pressed(minifb::Key::LeftBracket, minifb::KeyRepeat::No) {
            tracker.tuning.yaw_gain = (tracker.tuning.yaw_gain - 0.1).max(0.5);
            log::info!("yaw_gain {:.1}", tracker.tuning.yaw_gain);
        }
        if window.is_key_pressed(minifb::Key::Apostrophe, minifb::KeyRepeat::No) {
            tracker.tuning.pitch_gain = (tracker.tuning.pitch_gain + 0.1).min(6.0);
            log::info!("pitch_gain {:.1}", tracker.tuning.pitch_gain);
        }
        if window.is_key_pressed(minifb::Key::Semicolon, minifb::KeyRepeat::No) {
            tracker.tuning.pitch_gain = (tracker.tuning.pitch_gain - 0.1).max(0.5);
            log::info!("pitch_gain {:.1}", tracker.tuning.pitch_gain);
        }
        if window.is_key_pressed(minifb::Key::Period, minifb::KeyRepeat::No) {
            tracker.tuning.deadzone = (tracker.tuning.deadzone + 0.5).min(15.0);
            log::info!("deadzone {:.1}", tracker.tuning.deadzone);
        }
        if window.is_key_pressed(minifb::Key::Comma, minifb::KeyRepeat::No) {
            tracker.tuning.deadzone = (tracker.tuning.deadzone - 0.5).max(0.0);
            log::info!("deadzone {:.1}", tracker.tuning.deadzone);
        }
        if window.is_key_pressed(minifb::Key::M, minifb::KeyRepeat::No) {
            tracker.tuning.smoothing = (tracker.tuning.smoothing + 0.05).min(1.0);
            log::info!("smoothing {:.2}", tracker.tuning.smoothing);
        }
        if window.is_key_pressed(minifb::Key::N, minifb::KeyRepeat::No) {
            tracker.tuning.smoothing = (tracker.tuning.smoothing - 0.05).max(0.05);
            log::info!("smoothing {:.2}", tracker.tuning.smoothing);
        }
        if window.is_key_pressed(minifb::Key::I, minifb::KeyRepeat::No) {
            tracker.tuning.invert_yaw = !tracker.tuning.invert_yaw;
            log::info!("invert_yaw {}", tracker.tuning.invert_yaw);
        }
        if window.is_key_pressed(minifb::Key::O, minifb::KeyRepeat::No) {
            tracker.tuning.invert_pitch = !tracker.tuning.invert_pitch;
            log::info!("invert_pitch {}", tracker.tuning.invert_pitch);
        }
        if window.is_key_pressed(minifb::Key::P, minifb::KeyRepeat::No) {
            tracker.tuning.invert_roll = !tracker.tuning.invert_roll;
            log::info!("invert_roll {}", tracker.tuning.invert_roll);
        }
        // Toggle the experimental FaceMesh+Kabsch pose against the 2D one.
        if window.is_key_pressed(minifb::Key::Key3, minifb::KeyRepeat::No) {
            tracker.use_3d = !tracker.use_3d;
            tracker.recenter();
            log::info!(
                "pose {} (recentered)",
                if tracker.use_3d { "3D" } else { "2D" }
            );
            beep(BEEP_RECENTER);
        }

        // Guided capture: Space arms the sequence, a countdown follows, then
        // each stage runs for STAGE_SECS. `stage` is None while waiting.
        if record && armed.is_none() && window.is_key_pressed(minifb::Key::Space, minifb::KeyRepeat::No) {
            armed = Some(Instant::now());
            beep(BEEP_START);
            log::info!("armed; starting in {COUNTDOWN_SECS:.0}s");
        }
        let stage: Option<usize> = match (record, armed) {
            (true, Some(t0)) => {
                let t = t0.elapsed().as_secs_f32();
                (t >= COUNTDOWN_SECS)
                    .then(|| (((t - COUNTDOWN_SECS) / STAGE_SECS) as usize).min(RECORD_STAGES.len()))
            }
            _ => None,
        };
        if stage != last_stage {
            if let Some(s) = stage {
                if let Some(name) = RECORD_STAGES.get(s) {
                    log::info!("stage {}/{}: {name}", s + 1, RECORD_STAGES.len());
                    beep(BEEP_STAGE);
                }
            }
            last_stage = stage;
        }

        let frame = tracker.process(&rgb, width, height)?;

        if let Some(frame) = frame {
            draw_rect(&mut rgb, width, height, bx(frame.box_.0), bx(frame.box_.1), bx(frame.box_.2), bx(frame.box_.3), 255, 0, 0);

            let [eye_left, eye_right, nose, _mouth_r, _mouth_l] = frame.landmarks;
            draw_line(&mut rgb, width, height, eye_left, eye_right, 255, 255, 0);
            let eye_mid = (
                (eye_left.0 + eye_right.0) * 0.5,
                (eye_left.1 + eye_right.1) * 0.5,
            );
            draw_line(&mut rgb, width, height, eye_mid, nose, 0, 255, 255);
            for point in frame.landmarks {
                draw_dot(&mut rgb, width, height, point.0, point.1, 0, 255, 0);
            }

            if let (Some(rec), Some(s)) = (recorder.as_mut(), stage) {
                if s < RECORD_STAGES.len() {
                    let t = armed.map_or(0.0, |t0| t0.elapsed().as_secs_f32());
                    rec.row(t, s, &frame)?;
                }
            }

            if frames % TARGET_FPS as usize == 0 {
                log::info!(
                    "eye_dist={:4.1}px  yaw {:>6.1}  pitch {:>6.1}  roll {:>6.1}  [gain {:.1}/{:.1} dz {:.1} sm {:.2}]",
                    frame.eye_dist,
                    frame.pose.yaw,
                    frame.pose.pitch,
                    frame.pose.roll,
                    tracker.tuning.yaw_gain,
                    tracker.tuning.pitch_gain,
                    tracker.tuning.deadzone,
                    tracker.tuning.smoothing
                );
            }
        }

        // Mirror the camera image together with its face overlays, so the
        // preview behaves like a mirror. UI drawn below stays readable.
        if MIRROR_PREVIEW {
            mirror_rgb(&mut rgb, width, height);
        }

        if let Some(frame) = frame {
            let state = match (tracker.neutral.is_some(), tracker.use_3d) {
                (true, true) => "TRACKING 3D",
                (true, false) => "TRACKING",
                (false, true) => "UNCALIBRATED 3D",
                (false, false) => "UNCALIBRATED",
            };
            draw_hud(&mut rgb, width, height, state, frame.eye_dist, frame.pose, &tracker.tuning);
        }

        if record {
            match (armed, stage) {
                (None, _) => {
                    draw_text_shadow(
                        &mut rgb,
                        width,
                        height,
                        6,
                        56,
                        "PRESS SPACE TO START",
                        2,
                        255,
                        220,
                        0,
                    );
                    draw_sequence(&mut rgb, width, height, None);
                }
                (Some(t0), None) => {
                    let left = (COUNTDOWN_SECS - t0.elapsed().as_secs_f32()).max(0.0);
                    draw_text_shadow(
                        &mut rgb,
                        width,
                        height,
                        6,
                        56,
                        &format!("GET READY  {left:.0}"),
                        2,
                        255,
                        220,
                        0,
                    );
                    draw_sequence(&mut rgb, width, height, Some(0));
                }
                (Some(_), Some(s)) if s < RECORD_STAGES.len() => {
                    draw_guide(&mut rgb, width, height, stage_dir(s));
                    draw_text_shadow(
                        &mut rgb,
                        width,
                        height,
                        6,
                        56,
                        &format!(
                            "{}  {}/{}",
                            RECORD_STAGES[s].to_uppercase(),
                            s + 1,
                            RECORD_STAGES.len()
                        ),
                        2,
                        255,
                        220,
                        0,
                    );
                    draw_sequence(&mut rgb, width, height, Some(s));
                }
                _ => {}
            }
        }

        rgb_to_u32(&rgb, width, height, &mut buffer);
        window
            .update_with_buffer(&buffer, width, height)
            .context("failed to update the preview window")?;
        stats.sample(eye, read, work_start.elapsed());

        if stage == Some(RECORD_STAGES.len()) {
            log::info!("guided capture complete");
            break;
        }
    }
    Ok(())
}

/// Headless tracker with a small control panel: pose → opentrack over UDP.
///
/// Looks straight ahead at startup; recenters automatically after 3 s. A tiny
/// always-on-top panel has **Recenter** and **Quit** buttons; Ctrl+Alt+R also
/// recenters. Ctrl+C, `target/stop`, or closing the panel stop cleanly.
fn track_loop(eye: &Ps3Eye, width: usize, height: usize) -> Result<()> {
    let environment = ort::init()
        .with_telemetry(false)
        .build()
        .context("failed to initialise ONNX Runtime")?;
    let mut tracker = Tracker::load(
        Path::new(DETECTOR_MODEL),
        Path::new(LANDMARK_MODEL),
        &environment,
        output::DEFAULT_PORT,
    )?;
    tracker.use_3d |= has_flag("--3d");

    // Clean shutdown: Ctrl+C sets the flag, then the loop returns and the
    // caller stops the camera stream.
    let stop = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let stop = Arc::clone(&stop);
        move || stop.store(true, Ordering::SeqCst)
    })
    .context("failed to install the Ctrl+C handler")?;

    let stop_file = Path::new(STOP_FILE);
    let _ = std::fs::remove_file(stop_file);
    log::info!(
        "tracking; pose → opentrack at 127.0.0.1:{}",
        output::DEFAULT_PORT
    );
    log::info!("look straight ahead; recentering in 3 s");
    log::info!("stop with Ctrl+C or the control panel; Ctrl+Alt+R recenters");
    beep(BEEP_START);

    // Ctrl+Alt+R recenters from inside the game (global hotkey).
    let recenter = Arc::new(AtomicBool::new(false));
    hotkey::spawn(Arc::clone(&recenter)).context("failed to register Ctrl+Alt+R")?;

    // Small always-on-top control panel.
    let mut window = minifb::Window::new(
        "Headtrack",
        CONTROL_W,
        CONTROL_H,
        minifb::WindowOptions {
            topmost: true,
            resize: false,
            ..minifb::WindowOptions::default()
        },
    )
    .context("failed to open the control window")?;
    let mut control_rgb = vec![0_u8; CONTROL_W * CONTROL_H * 3];
    let mut buffer = vec![0_u32; CONTROL_W * CONTROL_H];
    let mut mouse_down = false;

    let started = Instant::now();
    let mut recentered = false;
    let mut frames: usize = 0;
    let mut stats = LoopStats::new();
    while !stop.load(Ordering::SeqCst) && !stop_file.exists() && window.is_open() {
        let read_start = Instant::now();
        let raw = eye.read_raw_frame().context("frame read failed")?;
        let read = read_start.elapsed();
        let work_start = Instant::now();
        frames += 1;
        let mut rgb = vec![0_u8; width * height * 3];
        bayer::bayer_grbg_to_rgb(&raw, width, height, &mut rgb);

        if let Some(frame) = tracker.process(&rgb, width, height)? {
            if !recentered && started.elapsed() >= Duration::from_secs(3) {
                tracker.recenter();
                recentered = true;
                log::info!("recentered; tracking active");
                beep(BEEP_RECENTER);
            }
            if recenter.swap(false, Ordering::SeqCst) {
                tracker.recenter();
                log::info!("recentered (Ctrl+Alt+R)");
                beep(BEEP_RECENTER);
            }
            if frames % TARGET_FPS as usize == 0 {
                log::info!(
                    "eye_dist={:4.1}px  yaw {:>6.1}  pitch {:>6.1}  roll {:>6.1}",
                    frame.eye_dist,
                    frame.pose.yaw,
                    frame.pose.pitch,
                    frame.pose.roll
                );
            }
        }

        let (on_recenter, on_quit) = control_hits(&window, &mut mouse_down);
        if on_recenter {
            tracker.recenter();
            log::info!("recentered (panel)");
            beep(BEEP_RECENTER);
        }
        if on_quit {
            break;
        }

        draw_control(
            &mut control_rgb,
            on_recenter,
            on_quit,
            tracker.neutral.is_some(),
        );
        rgb_to_u32(&control_rgb, CONTROL_W, CONTROL_H, &mut buffer);
        window
            .update_with_buffer(&buffer, CONTROL_W, CONTROL_H)
            .context("failed to update the control window")?;
        stats.sample(eye, read, work_start.elapsed());
    }

    log::info!("stopping; releasing the camera");
    Ok(())
}

/// Point-in-rectangle test.
fn in_rect(x: f32, y: f32, rx: usize, ry: usize, rw: usize, rh: usize) -> bool {
    x >= rx as f32 && x < (rx + rw) as f32 && y >= ry as f32 && y < (ry + rh) as f32
}

/// Read clicks on the control panel's buttons (edge-triggered).
fn control_hits(window: &minifb::Window, mouse_down: &mut bool) -> (bool, bool) {
    let down = window.get_mouse_down(minifb::MouseButton::Left);
    let edge = down && !*mouse_down;
    *mouse_down = down;
    if !edge {
        return (false, false);
    }
    let Some((mx, my)) = window.get_mouse_pos(minifb::MouseMode::Discard) else {
        return (false, false);
    };
    (
        in_rect(mx, my, RECENTER_X, BTN_Y, BTN_W, BTN_H),
        in_rect(mx, my, QUIT_X, BTN_Y, BTN_W, BTN_H),
    )
}

/// Fill a rectangle.
fn fill_rect(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    r: u8,
    g: u8,
    b: u8,
) {
    for yy in y..(y + h).min(height) {
        for xx in x..(x + w).min(width) {
            put_px(rgb, width, height, xx, yy, r, g, b);
        }
    }
}

/// Draw the control panel.
fn draw_control(rgb: &mut [u8], on_recenter: bool, on_quit: bool, tracking: bool) {
    let (w, h) = (CONTROL_W, CONTROL_H);
    fill_rect(rgb, w, h, 0, 0, w, h, 24, 24, 28);

    let status = if tracking { "TRACKING" } else { "UNCALIBRATED" };
    draw_text_shadow(rgb, w, h, 8, 8, status, 1, 200, 220, 255);

    let (rb, gb, bb) = if on_recenter {
        (80, 130, 200)
    } else {
        (40, 80, 140)
    };
    fill_rect(rgb, w, h, RECENTER_X, BTN_Y, BTN_W, BTN_H, rb, gb, bb);
    let (rq, gq, bq) = if on_quit {
        (200, 80, 80)
    } else {
        (140, 40, 40)
    };
    fill_rect(rgb, w, h, QUIT_X, BTN_Y, BTN_W, BTN_H, rq, gq, bq);

    let text_y = BTN_Y + (BTN_H - 7) / 2;
    let recenter_x = RECENTER_X + (BTN_W - "RECENTER".len() * 6) / 2;
    let quit_x = QUIT_X + (BTN_W - "QUIT".len() * 6) / 2;
    draw_text_shadow(rgb, w, h, recenter_x, text_y, "RECENTER", 1, 255, 255, 255);
    draw_text_shadow(rgb, w, h, quit_x, text_y, "QUIT", 1, 255, 255, 255);
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

/// Arrowhead at `tip`, pointing along the unit vector `dir`.
fn draw_head(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    tip: (f32, f32),
    dir: (f32, f32),
    c: (u8, u8, u8),
) {
    let (ux, uy) = dir;
    let head = 11.0_f32;
    for spread in [0.5_f32, -0.5] {
        let (s, k) = spread.sin_cos();
        let hx = tip.0 - head * (ux * k - uy * s);
        let hy = tip.1 - head * (uy * k + ux * s);
        draw_line(rgb, width, height, tip, (hx, hy), c.0, c.1, c.2);
    }
}

/// Straight arrow from `from` to `to`.
fn draw_arrow(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    from: (f32, f32),
    to: (f32, f32),
    c: (u8, u8, u8),
) {
    draw_line(rgb, width, height, from, to, c.0, c.1, c.2);
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let len = dx.hypot(dy).max(1.0);
    draw_head(rgb, width, height, to, (dx / len, dy / len), c);
}

/// Circular "turn your head" arrow centred on `centre`.
fn draw_rotate(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    centre: (f32, f32),
    radius: f32,
    clockwise: bool,
    c: (u8, u8, u8),
) {
    const SWEEP: f32 = 4.8; // radians, ~275°
    const STEPS: usize = 16;
    let start = 0.9_f32;
    let dir = if clockwise { 1.0_f32 } else { -1.0 };
    let at = |i: usize| start + dir * SWEEP * (i as f32 / STEPS as f32);
    let point = |a: f32| (centre.0 + radius * a.cos(), centre.1 + radius * a.sin());

    let mut prev = point(at(0));
    for i in 1..=STEPS {
        let p = point(at(i));
        draw_line(rgb, width, height, prev, p, c.0, c.1, c.2);
        prev = p;
    }
    let end = at(STEPS);
    draw_head(rgb, width, height, prev, (-end.sin() * dir, end.cos() * dir), c);
}

/// Six-way head-move guide: up, down, left, right and two roll arrows.
///
/// The arm for `active` is highlighted; the rest stay dim so the next move is
/// obvious at a glance.
fn draw_guide(rgb: &mut [u8], width: usize, height: usize, active: Option<Dir>) {
    let cx = width as f32 * 0.5;
    let cy = height as f32 * 0.5 - 30.0;
    let arm = 96.0_f32;
    let colour = |dir: Dir| {
        if active == Some(dir) {
            (255, 220, 0)
        } else {
            (110, 110, 110)
        }
    };

    draw_arrow(rgb, width, height, (cx, cy - 16.0), (cx, cy - arm), colour(Dir::Up));
    draw_arrow(rgb, width, height, (cx, cy + 16.0), (cx, cy + arm), colour(Dir::Down));
    draw_arrow(rgb, width, height, (cx - 16.0, cy), (cx - arm, cy), colour(Dir::Left));
    draw_arrow(rgb, width, height, (cx + 16.0, cy), (cx + arm, cy), colour(Dir::Right));

    draw_rotate(rgb, width, height, (cx - 52.0, cy + 52.0), 26.0, false, colour(Dir::RollLeft));
    draw_rotate(rgb, width, height, (cx + 52.0, cy + 52.0), 26.0, true, colour(Dir::RollRight));
}

/// The whole capture sequence, numbered, with the current stage highlighted.
fn draw_sequence(rgb: &mut [u8], width: usize, height: usize, active: Option<usize>) {
    let mut y = 92_usize;
    for (index, name) in RECORD_STAGES.iter().enumerate() {
        let c = if active == Some(index) {
            (255, 220, 0)
        } else {
            (150, 150, 150)
        };
        let label = format!("{}. {}", index + 1, name.to_uppercase());
        draw_text_shadow(rgb, width, height, 8, y, &label, 1, c.0, c.1, c.2);
        y += 18;
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
        "N M SMOOTH   I O P INVERT   3 2D/3D   ESC QUIT",
        1,
        220,
        220,
        220,
    );
}

/// Flip an RGB8 image horizontally, in place.
#[expect(
    clippy::indexing_slicing,
    reason = "pixel offsets are within the validated frame buffer"
)]
fn mirror_rgb(rgb: &mut [u8], width: usize, height: usize) {
    for y in 0..height {
        let row = y * width * 3;
        for x in 0..width / 2 {
            let a = row + x * 3;
            let b = row + (width - 1 - x) * 3;
            for i in 0..3 {
                rgb.swap(a + i, b + i);
            }
        }
    }
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
