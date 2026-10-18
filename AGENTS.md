# AGENTS.md — Headtrack

Instructions for AI coding agents (and humans) working in this repository.

## Repo identity

- Public name: **PS3EYE ETS2 Headtracker**
- GitHub: `menschenfracht/ps3eye-ets2-headtracker` (to be created/pushed)
- License: **GPL-2.0-only** (`LICENSE` at repo root, `license` in Cargo.toml)
  — strict, because the OV534/OV772x register tables and payload scanner are
  derived from the Linux kernel `gspca-ov534` driver (GPLv2-only), via
  PS3EYEDriver (MIT new code / GPLv2 kernel-derived parts). Do NOT change to
  a permissive license; do NOT remove the credits section in `README.md`.
- `docs/reference/` holds third-party source for porting. It is **gitignored**
  — never commit it, never republish it.

## Privacy rules — hard requirements

- Never commit images from camera testing (`docs/*.jpg|png|ppm` — they show
  the user's face). `.gitignore` enforces this; keep it that way.
- Never commit the terminal screenshot in the WSL home.
- Before `git push`, sanity-check `git status` for stray media files.

## What this is

A DIY head tracker for Euro Truck Simulator 2 (and other sims), built in Rust.
Hardware: **Sony PlayStation 3 Eye** camera. Goal: rough head tracking
(Tobii-like) — camera → head pose → smoothing → TrackIR-compatible output.

## Layout

```
D:\_coding\Headtrack\
├── apps\
│   ├── PS3EyeFilter\   DirectShow source filter (makes the PS3 Eye a webcam)
│   ├── AITrack\        reference AI tracker (fallback / comparison)
│   └── opentrack\      smoothing + TrackIR output bridge for ETS2
├── src\                the Rust tracker crate (ours)
├── docs\               protocol notes, calibration records, research
└── AGENTS.md           this file
```

## Environment quirks — READ BEFORE RUNNING ANYTHING

- The agent session runs in **WSL2 (Ubuntu)**. Windows tools are driven via
  `cmd.exe` / `powershell.exe` from `/mnt/c/Windows/System32/...`.
- **CMD from WSL starts in an unsupported UNC path** (`\\wsl.localhost\...`) and
  prints a warning, defaulting to `C:\Windows`. ALWAYS pin the working
  directory explicitly:
  - CMD: `cmd /c "cd /d D:\_coding\Headtrack\src && cargo build"`
  - PowerShell: `Set-Location D:\_coding\Headtrack\src; ...`
- Admin operations require a **UAC prompt on the user's screen**: write a
  `.ps1` script, launch with
  `Start-Process powershell -Verb RunAs -Wait -ArgumentList '...'`,
  have the script `Start-Transcript` to a log file, then read the log.
- The Windows side has its own **rustup/cargo (1.99)**. Build and run the
  tracker from the Windows side only (camera + ETS2 are Windows-only).
  Never assume the WSL rustup is the same toolchain.
- Prefer `D:\_coding\Headtrack\...` paths: no spaces, no Cyrillic, no OneDrive.

## Hardware / driver facts (as of 2026-10-18)

- Camera: **PS3 Eye** — `USB\VID_1415&PID_2000` (Nam Tai / OmniVision, OV534).
  NOT a UVC device; it needs the libusb-family driver + a DirectShow filter.
- Interface 0 = video (bound to **libusbK**, `oem56.inf`, class
  "libusbK Usb Devices"). Interface 1 = mic array (Microsoft inbox
  `wdma_usb.inf` / `usbaudio`). Both must show ProblemCode 0.
- The old deprecated **libusb-win32 1.2.6.0** driver was fully uninstalled
  (driver store, service, self-signed certs). Do NOT reinstall it.
- Driver package backup: `C:\USBCAM-libusbK\` (regenerate with
  `libusbK-inf-wizard.exe` from `~/webcam win drivers/libusbK-dev-kit/`).
- The camera firmware **gets stuck** if apps exit uncleanly or USB power
  management kicks in. Recovery: `pnputil /disable-device` +
  `/enable-device` on the parent instance — NOT replugging (replugs spawn
  phantom device instances).
- USB selective suspend is disabled system-wide (power plan). Keep it that
  way. The libusbK INF was generated with device idle power management
  disabled on purpose.
- The camera sits on the Intel USB 3.0 xHCI controller (port 2-12); PS3 Eyes
  are known to misbehave on xHCI — if lockups persist, try a USB 2.0 hub.
- Camera has a **manual focus ring** with two detents (~30 cm / ~1 m).
- The camera is IR-sensitive — optional night-mode later via IR illuminator.

## Rust conventions

Style follows the user's `bevy_tut` project:

- **Edition 2024**, current stable toolchain.
- Clippy contract in `Cargo.toml` (`[lints.clippy]`): `all`, `nursery`,
  `cargo`, `perf`, `pedantic` = warn; `correctness` = deny; plus the same
  explicit allowances/preferences as bevy_tut (see that project's Cargo.toml
  — `indexing_slicing` deny, `explicit_counter_loop` deny, `implicit_return`
  allow, `print_stdout`/`print_stderr` warn, `allow_attributes_without_reason`
  warn, etc.). Copy the block verbatim when scaffolding.
- `#[expect(clippy::..., reason = "...")]` over `#[allow]`.
- Release profile: `opt-level 3`, `lto = "thin"`, `codegen-units = 1`,
  `panic = "abort"`, `strip = true`.
- Modules split by concern (capture / inference / filter / output / preview),
  state machines over flag soup, doc comments on public items.

## Pipeline / architecture

```
nokhwa (DirectShow, 320x240@60..120) → blazeface ONNX + 6D-pose ONNX via
`ort` → One-Euro filter → output
```

Output, in order of appearance:
1. **MVP**: UDP → opentrack (which drives ETS2 via TrackIR protocol).
2. **V2**: implement the TrackIR/NPClient protocol natively, no opentrack.

## Milestones

- [ ] M1 — frames from the PS3 Eye into Rust (nokhwa), printed FPS
- [ ] M2 — head pose live (ONNX via `ort`), egui preview with landmarks
- [ ] M3 — One-Euro smoothing + UDP out; verified against opentrack
- [ ] M4 — first drive in ETS2 (TrackIR enabled, `trackIR 1` if needed)
- [ ] M5 — native TrackIR protocol (drop opentrack); optional IR night mode

## Arithmetic and conversion rules

- No bare `as` casts between numeric types. Use `T::try_from(x)` and handle the error, or a `From` impl when the widening is lossless.
- Do arithmetic in a wider type before narrowing: sum in `u16` before halving a pair of `u8`, not after truncating to `u8`.
- Narrow with `try_from(...).unwrap_or(MAX)` (or another explicit fallback) on hot paths; `expect` only where the value is an invariant with a stated reason.
- Don't hide overflow with clamping (`saturating_*`, `clamp`) unless the saturation is the intended behavior and is noted at the call site.
- Pointer and FFI casts (`as *mut T`, `.cast()`) are exempt, but each unsafe block needs a SAFETY comment.

## House rules

- Don't touch anything outside `D:\_coding\Headtrack\` and the documented
  driver backup paths without asking.
- `webcam win drivers\` in the WSL home is a **driver archive** — read-only.
- Keep tools portable where possible; document any registry change in `docs\`.
- When changing driver/USB state, verify with
  `Get-PnpDevice | ? InstanceId -like '*VID_1415*'` afterwards and report
  Status/ProblemCode for all three instances (parent, MI_00, MI_01).
