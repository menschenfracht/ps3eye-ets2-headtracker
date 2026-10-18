//! Low-level OV534 bridge and OV772x sensor register access.
//!
//! Ported from `PS3EYEDriver` (MIT, © Eugene Zatepyakin) and the original
//! gspca-ov534 kernel driver. The bridge is commanded over USB control
//! transfers; the image sensor sits behind it on an SCCB bus.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use rusb::{DeviceHandle, UsbContext};

/// Vendor request for both register reads and writes.
const OV534_REQ_REG: u8 = 0x01;
/// SCCB bridge registers.
const REG_SUBADDR: u16 = 0xf2;
const REG_WRITE: u16 = 0xf3;
const REG_READ: u16 = 0xf4;
const REG_OPERATION: u16 = 0xf5;
const REG_STATUS: u16 = 0xf6;
/// SCCB operation codes.
const OP_WRITE_3: u8 = 0x37;
const OP_WRITE_2: u8 = 0x33;
const OP_READ_2: u8 = 0xf9;

const CONTROL_TIMEOUT: Duration = Duration::from_millis(500);

/// Register-level access to the OV534 bridge and the sensor behind it.
pub struct Bridge<T: UsbContext> {
    handle: DeviceHandle<T>,
}

impl<T: UsbContext> Bridge<T> {
    /// Wraps an already-opened device handle.
    pub const fn new(handle: DeviceHandle<T>) -> Self {
        Self { handle }
    }

    /// Consume the bridge, returning the underlying handle.
    #[must_use]
    pub fn into_handle(self) -> DeviceHandle<T> {
        self.handle
    }

    /// Write one bridge register.
    pub fn reg_write(&self, reg: u16, val: u8) -> Result<()> {
        let n = self
            .handle
            .write_control(0x40, OV534_REQ_REG, 0x0000, reg, &[val], CONTROL_TIMEOUT)
            .context("ov534 register write failed")?;
        if n != 1 {
            bail!("ov534 register write wrote {n} byte(s), expected 1");
        }
        Ok(())
    }

    /// Read one bridge register.
    pub fn reg_read(&self, reg: u16) -> Result<u8> {
        let mut buf = [0_u8; 1];
        let n = self
            .handle
            .read_control(0xC0, OV534_REQ_REG, 0x0000, reg, &mut buf, CONTROL_TIMEOUT)
            .context("ov534 register read failed")?;
        if n != 1 {
            bail!("ov534 register read got {n} byte(s), expected 1");
        }
        Ok(buf[0])
    }

    /// Poll the SCCB engine until the last operation succeeded.
    fn sccb_check_status(&self) -> Result<()> {
        for attempt in 1..=5 {
            match self.reg_read(REG_STATUS) {
                Ok(0x00) => return Ok(()),
                Ok(0x04) => bail!("sccb operation failed (status 0x04)"),
                Ok(0x03) => {} // busy, retry
                Ok(other) => {
                    log::debug!("sccb status 0x{other:02x}, attempt {attempt}/5");
                }
                Err(err) => return Err(err).with_context(|| format!("sccb status poll {attempt}")),
            }
        }
        bail!("sccb status did not settle after 5 attempts")
    }

    /// Write one sensor register through the SCCB bridge.
    pub fn sccb_write(&self, reg: u8, val: u8) -> Result<()> {
        self.reg_write(REG_SUBADDR, reg)?;
        self.reg_write(REG_WRITE, val)?;
        self.reg_write(REG_OPERATION, OP_WRITE_3)?;
        self.sccb_check_status()
    }

    /// Read one sensor register through the SCCB bridge.
    pub fn sccb_read(&self, reg: u8) -> Result<u8> {
        self.reg_write(REG_SUBADDR, reg)?;
        self.reg_write(REG_OPERATION, OP_WRITE_2)?;
        self.sccb_check_status()?;

        self.reg_write(REG_OPERATION, OP_READ_2)?;
        self.sccb_check_status()?;

        self.reg_read(REG_READ)
    }

    /// Run a `(register, value)` table against the bridge.
    pub fn write_bridge_table(&self, table: &[[u8; 2]]) -> Result<()> {
        for &[reg, val] in table {
            self.reg_write(u16::from(reg), val)?;
        }
        Ok(())
    }

    /// Run a `(register, value)` table against the sensor.
    /// A `0xff` entry reads the register listed in the value (quirk kept
    /// from the reference driver), then clears it.
    pub fn write_sensor_table(&self, table: &[[u8; 2]]) -> Result<()> {
        for &[reg, val] in table {
            if reg != 0xff {
                self.sccb_write(reg, val)?;
            } else {
                self.sccb_read(val)?;
                self.sccb_write(0xff, 0x00)?;
            }
        }
        Ok(())
    }

    /// Turn the blue LED on or off.
    pub fn set_led(&self, on: bool) -> Result<()> {
        let mut data = self.reg_read(0x21)?;
        data |= 0x80;
        self.reg_write(0x21, data)?;

        data = self.reg_read(0x23)?;
        if on {
            data |= 0x80;
        } else {
            data &= 0x7f;
        }
        self.reg_write(0x23, data)?;

        if !on {
            data = self.reg_read(0x21)?;
            data &= 0x7f;
            self.reg_write(0x21, data)?;
        }
        Ok(())
    }
}

/// Frame rate table entry: `(fps, sensor 0x11, sensor 0x0d, bridge 0xe5)`.
struct Rate {
    fps: u16,
    r11: u8,
    r0d: u8,
    re5: u8,
}

const RATES_VGA: [Rate; 14] = [
    Rate { fps: 83, r11: 0x01, r0d: 0xc1, re5: 0x02 }, // partly corrupt
    Rate { fps: 75, r11: 0x01, r0d: 0x81, re5: 0x02 },
    Rate { fps: 60, r11: 0x00, r0d: 0x41, re5: 0x04 },
    Rate { fps: 50, r11: 0x01, r0d: 0x41, re5: 0x02 },
    Rate { fps: 40, r11: 0x02, r0d: 0xc1, re5: 0x04 },
    Rate { fps: 30, r11: 0x04, r0d: 0x81, re5: 0x02 },
    Rate { fps: 25, r11: 0x00, r0d: 0x01, re5: 0x02 },
    Rate { fps: 20, r11: 0x04, r0d: 0x41, re5: 0x02 },
    Rate { fps: 15, r11: 0x09, r0d: 0x81, re5: 0x02 },
    Rate { fps: 10, r11: 0x09, r0d: 0x41, re5: 0x02 },
    Rate { fps: 8, r11: 0x02, r0d: 0x01, re5: 0x02 },
    Rate { fps: 5, r11: 0x04, r0d: 0x01, re5: 0x02 },
    Rate { fps: 3, r11: 0x06, r0d: 0x01, re5: 0x02 },
    Rate { fps: 2, r11: 0x09, r0d: 0x01, re5: 0x02 },
];

const RATES_QVGA: [Rate; 22] = [
    Rate { fps: 290, r11: 0x00, r0d: 0xc1, re5: 0x04 }, // partly corrupt
    Rate { fps: 205, r11: 0x01, r0d: 0xc1, re5: 0x02 }, // partly corrupt
    Rate { fps: 187, r11: 0x01, r0d: 0x81, re5: 0x02 },
    Rate { fps: 150, r11: 0x00, r0d: 0x41, re5: 0x04 },
    Rate { fps: 137, r11: 0x02, r0d: 0xc1, re5: 0x02 },
    Rate { fps: 125, r11: 0x01, r0d: 0x41, re5: 0x02 },
    Rate { fps: 100, r11: 0x02, r0d: 0xc1, re5: 0x04 },
    Rate { fps: 90, r11: 0x03, r0d: 0x81, re5: 0x02 },
    Rate { fps: 75, r11: 0x04, r0d: 0x81, re5: 0x02 },
    Rate { fps: 60, r11: 0x04, r0d: 0xc1, re5: 0x04 },
    Rate { fps: 50, r11: 0x04, r0d: 0x41, re5: 0x02 },
    Rate { fps: 40, r11: 0x06, r0d: 0x81, re5: 0x03 },
    Rate { fps: 37, r11: 0x00, r0d: 0x01, re5: 0x04 },
    Rate { fps: 30, r11: 0x04, r0d: 0x41, re5: 0x04 },
    Rate { fps: 17, r11: 0x18, r0d: 0xc1, re5: 0x02 },
    Rate { fps: 15, r11: 0x18, r0d: 0x81, re5: 0x02 },
    Rate { fps: 12, r11: 0x02, r0d: 0x01, re5: 0x04 },
    Rate { fps: 10, r11: 0x18, r0d: 0x41, re5: 0x02 },
    Rate { fps: 7, r11: 0x04, r0d: 0x01, re5: 0x04 },
    Rate { fps: 5, r11: 0x06, r0d: 0x01, re5: 0x04 },
    Rate { fps: 3, r11: 0x09, r0d: 0x01, re5: 0x04 },
    Rate { fps: 2, r11: 0x18, r0d: 0x01, re5: 0x02 },
];

/// Pick the closest supported frame rate and program it.
///
/// With `dry_run` the value is only validated and returned.
pub fn set_frame_rate<T: UsbContext>(
    bridge: &Bridge<T>,
    width: u32,
    desired: u16,
    dry_run: bool,
) -> Result<u16> {
    let table = if width == 640 { &RATES_VGA } else { &RATES_QVGA };
    let rate = table
        .iter()
        .find(|rate| desired >= rate.fps)
        .with_context(|| format!("no frame rate at or below {desired} fps"))?;

    if !dry_run {
        bridge.sccb_write(0x11, rate.r11)?;
        bridge.sccb_write(0x0d, rate.r0d)?;
        bridge.reg_write(0xe5, rate.re5)?;
    }
    Ok(rate.fps)
}

/// Bridge registers to run once after power-up (before sensor init).
pub const BRIDGE_INIT: &[[u8; 2]] = &[
    [0xe7, 0x3a],
    [0xf1, 0x42], // select the OV772x sensor
    [0x92, 0x01],
    [0x93, 0x18],
    [0x94, 0x10],
    [0x95, 0x10],
    [0xe2, 0x00],
    [0xe7, 0x3e],
    [0x96, 0x00],
    [0x97, 0x20],
    [0x97, 0x20],
    [0x97, 0x20],
    [0x97, 0x0a],
    [0x97, 0x3f],
    [0x97, 0x4a],
    [0x97, 0x20],
    [0x97, 0x15],
    [0x97, 0x0b],
    [0x8e, 0x40],
    [0x1f, 0x81],
    [0xc0, 0x50],
    [0xc1, 0x3c],
    [0xc2, 0x01],
    [0xc3, 0x01],
    [0x50, 0x89],
    [0x88, 0x08],
    [0x8d, 0x00],
    [0x8e, 0x00],
    // video data start (V_FMT)
    [0x1c, 0x00],
    // RAW8 mode, payload size 0x0200 * 4 = 2048 bytes
    [0x1d, 0x00],
    [0x1d, 0x02],
    [0x1d, 0x00],
    // frame size = 0x012c00 * 4 = 307200 bytes (640*480 @ 8bpp)
    [0x1d, 0x01],
    [0x1d, 0x2c],
    [0x1d, 0x00],
    // video data start (V_CNTL0), turn on UVC header
    [0x1c, 0x0a],
    [0x1d, 0x08],
    [0x1d, 0x0e],
    [0x34, 0x05],
    [0xe3, 0x04],
    [0x89, 0x00],
    [0x76, 0x00],
    [0xe7, 0x2e],
    [0x31, 0xf9],
    [0x25, 0x42],
    [0x21, 0xf0],
    [0xe5, 0x04],
];

/// Sensor registers to run once after power-up.
pub const SENSOR_INIT: &[[u8; 2]] = &[
    [0x12, 0x80], // reset
    [0x3d, 0x00],
    [0x12, 0x01], // processed Bayer RAW (8 bit)
    [0x11, 0x01],
    [0x14, 0x40],
    [0x15, 0x00],
    [0x63, 0xaa], // auto white balance
    [0x64, 0x87],
    [0x66, 0x00],
    [0x67, 0x02],
    [0x17, 0x26],
    [0x18, 0xa0],
    [0x19, 0x07],
    [0x1a, 0xf0],
    [0x29, 0xa0],
    [0x2a, 0x00],
    [0x2c, 0xf0],
    [0x20, 0x10],
    [0x4e, 0x0f],
    [0x3e, 0xf3],
    [0x0d, 0x41],
    [0x32, 0x00],
    [0x13, 0xf0], // COM8
    [0x22, 0x7f],
    [0x23, 0x03],
    [0x24, 0x40],
    [0x25, 0x30],
    [0x26, 0xa1],
    [0x2a, 0x00],
    [0x2b, 0x00],
    [0x13, 0xf7],
    [0x0c, 0xc0],
    [0x11, 0x00],
    [0x0d, 0x41],
    [0x8e, 0x00], // de-noise threshold
];

/// Bridge registers to start VGA (640×480) streaming.
pub const BRIDGE_START_VGA: &[[u8; 2]] = &[
    [0x1c, 0x00],
    [0x1d, 0x00],
    [0x1d, 0x02],
    [0x1d, 0x00],
    [0x1d, 0x01], // frame size = 0x012c00 * 4 = 307200 bytes
    [0x1d, 0x2c],
    [0x1d, 0x00],
    [0xc0, 0x50],
    [0xc1, 0x3c],
];

/// Sensor registers to start VGA (640×480) streaming.
pub const SENSOR_START_VGA: &[[u8; 2]] = &[
    [0x12, 0x01],
    [0x17, 0x26],
    [0x18, 0xa0],
    [0x19, 0x07],
    [0x1a, 0xf0],
    [0x29, 0xa0],
    [0x2c, 0xf0],
    [0x65, 0x20],
];

/// Bridge registers to start QVGA (320×240) streaming.
pub const BRIDGE_START_QVGA: &[[u8; 2]] = &[
    [0x1c, 0x00],
    [0x1d, 0x00],
    [0x1d, 0x02],
    [0x1d, 0x00],
    [0x1d, 0x00], // frame size = 0x004b00 * 4 = 76800 bytes
    [0x1d, 0x4b],
    [0x1d, 0x00],
    [0xc0, 0x28],
    [0xc1, 0x1e],
];

/// Sensor registers to start QVGA (320×240) streaming.
pub const SENSOR_START_QVGA: &[[u8; 2]] = &[
    [0x12, 0x41],
    [0x17, 0x3f],
    [0x18, 0x50],
    [0x19, 0x03],
    [0x1a, 0x78],
    [0x29, 0x50],
    [0x2c, 0x78],
    [0x65, 0x2f],
];
