# PS3EYE ETS2 Headtracker

A DIY head tracker for **Euro Truck Simulator 2** (and other sims), built in
Rust on top of the **Sony PlayStation 3 Eye** camera — a Tobii-like look-around
for your truck cab, using hardware you already have in a drawer.

```
PS3 Eye (OV534) ──libusbK──► headtrack (this crate) ──UDP──► opentrack ──► ETS2
                              capture → face → head pose → smoothing
```

## Why the PS3 Eye?

It was *the* head-tracking camera a decade ago, and it is still excellent:
manual focus, an IR-sensitive sensor, and — thanks to its non-UVC OV534
bridge — a hardware uniqueness that forces you to write your own driver, which
is half the fun.

## Status

Working and driven daily; this is a personal DIY project, not a product.

- [x] Camera identification and driver setup (libusbK)
- [x] M1.0 — USB enumeration through `rusb`
- [x] M1.1 — OV534/OV772x protocol port, bulk streaming at 640×480
- [x] M2 — head pose (YuNet + a 2D landmark heuristic), preview window
- [x] M3 — One-Euro smoothing + UDP output
- [x] M4 — driving in ETS2 through opentrack
- [ ] M5 — native TrackIR protocol (drop opentrack)
- [ ] Experimental — 3D pose from FaceMesh landmarks (parked, `HEADTRACK_3D`)

## Hardware & driver requirements

- Sony PS3 Eye (`USB VID_1415 PID_2000`)
- The camera is **not** UVC — it needs the [libusbK](https://github.com/libusbK/libusbK)
  driver bound to interface 0 (generate with `libusbK-inf-wizard.exe`, see
  [opentrack's PS3 Eye open-driver instructions](https://github.com/opentrack/opentrack/wiki/PS3-Eye-open-driver-instructions)).
  The camera's microphone (interface 1) can stay on the Microsoft driver.
- [opentrack](https://github.com/opentrack/opentrack) 2026.1.0 or newer.

### Models

`models/face_detection_yunet.onnx` (YuNet, MIT) is the only model the default
pipeline needs. `models/face_landmark_192.onnx` (FaceMesh) is used only by the
experimental 3D pose and the `--facemesh` harness.

## Building

```sh
cargo build --release
```

Run it from the repository root: the models are resolved relative to the
working directory (`models/…`), and logs/captures go to `docs/`.

## Running

| Mode | What it does |
|---|---|
| `headtrack.exe` | stream only, reports the measured frame rate |
| `headtrack.exe --preview` | live mirrored preview with overlays and tuning keys |
| `headtrack.exe --track` | headless: pose → opentrack over UDP, tiny control panel |
| `headtrack.exe --record` | guided capture (see below), writes `docs/pose.csv` |
| `headtrack.exe --preview --log` | preview plus per-frame `docs/live.csv` |
| `headtrack.exe --detect` | detector bench, saves annotated frames |
| `headtrack.exe --capture` | saves clean frames for offline analysis |
| `headtrack.exe --facemesh` | FaceMesh inspection harness |

`--track` is the normal way to use it in game. It recenters 3 s after it first
sees a face, then responds to the recenter hotkey, `target/stop`, Ctrl+C, or
the panel's Quit button. `--3d` (or the `3` key in preview) switches to the
experimental FaceMesh pose.

### Preview keys

`R` recenter · `[` `]` yaw gain · `;` `'` pitch gain · `,` `.` deadzone ·
`N` `M` smoothing · `I` `O` `P` invert yaw/pitch/roll · `3` 2D/3D pose · `Esc` quit

### Guided capture

`--record` shows a six-way cross (up, down, left, right, tilt left, tilt right)
plus the full numbered sequence. Press **Space** to arm it, wait out the 3 s
countdown, then follow the highlighted arrow for 4 s per pose. Every frame is
written to `docs/pose.csv` with the landmarks, both poses, and a `detected`
flag, which is what the pitch estimator was designed against.

## opentrack setup

1. **Input:** Tracker → *FreePIE UDP receiver*, port `5555`.
2. **Output:** Protocol → *freetrack 2.0 Enhanced*, then start the tracker in
   ETS2's TrackIR slot.
3. `invert_yaw` is already applied by `headtrack`, so the angles arrive in
   opentrack's convention. Use opentrack's mapping tab if you want more throw
   (e.g. multiply yaw).
4. **Quit quirk:** in 2026.1.0, closing opentrack with an active
   *video-feed-less* tracker trips `assert(opt->receivers.empty())`
   ([opentrack#2130](https://github.com/opentrack/opentrack/issues/2130)).
   Click **Stop** before closing the window and it exits cleanly.

## How the pose is estimated

Five YuNet landmarks (both eyes, nose tip, both mouth corners) give:

- **roll** — the angle of the eye line;
- **yaw** — the nose offset along the eye line, over the eye distance;
- **pitch** — where the nose sits between the eye line and the mouth line.

The nose and mouth offsets are rotated into the face frame first, so a head
tilt does not leak into pitch. Pitch is a *ratio* of two distances along the
same direction, which cancels the face's apparent size — important because the
eye distance collapses sharply when you turn your head.

A single frame costs about 15 ms at 640×480, so the pipeline runs at **30 fps**
with roughly one frame of latency. If the reader falls behind, it drops to the
newest queued frame rather than accumulating delay.

### Recenter geometry

The camera sits on top of the monitor, so looking at the lens is several
degrees *above* looking at the screen centre — the pose the game should treat
as zero. Recentring on the camera therefore biases the neutral down by that
angle, computed from the rig constants at the top of `src/tracker.rs`:

```rust
MONITOR_DIAGONAL_IN  = 32.0   // inches, 16:9
MONITOR_ASPECT       = (16.0, 9.0)
EYE_TO_SCREEN_CM     = 66.0
CAMERA_ABOVE_EDGE_CM = 1.0
EYE_ABOVE_CENTRE_CM  = 10.0   // eyes above the screen centre
```

Change those to match your desk and the correction follows. The startup log
prints the resulting angle.

## Cues

| Event | Sound |
|---|---|
| startup | 1000 Hz, 250 ms |
| recenter (any trigger) | 1500 Hz, 100 ms |
| capture stage change | 2000 Hz, 60 ms |
| face lost > 1 s | falling 900 / 800 / 700 Hz, repeated every 4 s |

## License

**GPL-2.0-only** — see [LICENSE](LICENSE).

The OV534/OV772x register tables and the payload scanner are derived from the
Linux kernel `gspca-ov534` driver (GPL-2.0-only, © Jean-François Moine,
Antonio Ospite and contributors), via
[PS3EYEDriver](https://github.com/inspirit/PS3EYEDriver) (© Eugene Zatepyakin,
MIT for new code, GPLv2 for kernel-derived parts). GPL-2.0-only is therefore
the strictest license we can honestly pick for the combined work.

## Credits

- [PS3EYEDriver](https://github.com/inspirit/PS3EYEDriver) — Eugene Zatepyakin
- [gspca ov534](https://www.kernel.org/) — Linux kernel driver authors
- [libusbK](https://github.com/libusbK/libusbK) — Travis Lee Robinson and the libusb community
- [opentrack](https://github.com/opentrack/opentrack) — the head-tracking ecosystem this plugs into
- [YuNet](https://github.com/opencv/opencv_zoo) — OpenCV Zoo face detector (MIT)
- [AllanCat/PS3EyeDirectShow](https://github.com/AllanCat/PS3EyeDirectShow) and
  [visapex-app/PS3EyeDirectShow-120fps](https://github.com/visapex-app/PS3EyeDirectShow-120fps)
