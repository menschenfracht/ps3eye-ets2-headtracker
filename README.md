# PS3EYE ETS2 Headtracker

A DIY head tracker for **Euro Truck Simulator 2** (and other sims), built in
Rust on top of the **Sony PlayStation 3 Eye** camera — a Tobii-like look-around
for your truck cab, using hardware you already have in a drawer.

```
PS3 Eye (OV534) ──libusbK──► headtrack (this crate) ──UDP──► opentrack ──► ETS2
                              capture → head pose → smoothing
```

## Why the PS3 Eye?

It was *the* head-tracking camera a decade ago, and it is still excellent:
120 fps at 320×240, manual focus, IR-sensitive sensor, and — thanks to its
non-UVC OV534 bridge — a hardware uniqueness that forces us to write our own
driver, which is half the fun.

## Status

Work in progress. See `AGENTS.md` for the milestone list.

- [x] Camera identification and driver setup (libusbK)
- [x] DirectShow filter for using the camera in regular apps
- [x] M1.0 — USB enumeration through `rusb`
- [ ] M1.1 — OV534/OV772x protocol port, bulk streaming
- [ ] M2 — head pose (ONNX), preview window
- [ ] M3 — One-Euro smoothing + UDP output
- [ ] M4 — first drive in ETS2
- [ ] M5 — native TrackIR protocol (drop opentrack)

## Hardware & driver requirements

- Sony PS3 Eye (`USB VID_1415 PID_2000`)
- The camera is **not** UVC — it needs the [libusbK](https://github.com/libusbK/libusbK)
  driver bound to interface 0 (generate with `libusbK-inf-wizard.exe`, see
  [opentrack's PS3 Eye open-driver instructions](https://github.com/opentrack/opentrack/wiki/PS3-Eye-open-driver-instructions))
- Optional (for AITrack/opentrack comparison runs): a PS3 Eye DirectShow
  filter, e.g. [PS3EyeDirectShow-120fps](https://github.com/visapex-app/PS3EyeDirectShow-120fps)

## Building

```sh
cd src
cargo build --release
```

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
- [AllanCat/PS3EyeDirectShow](https://github.com/AllanCat/PS3EyeDirectShow) and
  [visapex-app/PS3EyeDirectShow-120fps](https://github.com/visapex-app/PS3EyeDirectShow-120fps)
