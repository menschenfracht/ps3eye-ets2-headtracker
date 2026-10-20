//! The PS3 Eye camera: lifecycle, OV534 bring-up, and frame streaming.
//!
//! Ported from `PS3EYEDriver` (MIT for new code / GPLv2 for kernel-derived
//! parts; see README). Streaming uses an asynchronous bulk URB ring, the same
//! design as the reference driver (see `async_stream`).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{ensure, Context, Result};

use crate::async_stream::AsyncStream;
use crate::ov534::{self, Bridge, SharedHandle};
use crate::scanner::PktScanner;

/// Nam Tai E&E / OmniVision — the PS3 Eye's USB identifiers.
const PS3_EYE_VID: u16 = 0x1415;
const PS3_EYE_PID: u16 = 0x2000;

/// How long to wait for the next frame before giving up.
const FRAME_TIMEOUT: Duration = Duration::from_secs(5);

/// Supported output resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    Vga,
    Qvga,
}

impl Resolution {
    #[must_use]
    pub const fn dims(self) -> (u32, u32) {
        match self {
            Self::Vga => (640, 480),
            Self::Qvga => (320, 240),
        }
    }

    #[must_use]
    pub const fn frame_size(self) -> usize {
        match self {
            Self::Vga => 640 * 480,
            Self::Qvga => 320 * 240,
        }
    }
}

/// An opened, initialized PS3 Eye.
pub struct Ps3Eye {
    bridge: Bridge,
    bulk_endpoint: u8,
    resolution: Option<Resolution>,
    sensor_id: u16,
    fps: u16,
    streaming: bool,
    stream: Option<AsyncStream>,
}

impl Ps3Eye {
    /// Find the first PS3 Eye on the bus, open it, and claim interface 0.
    pub fn open() -> Result<Self> {
        let devices = rusb::DeviceList::new().context("failed to list USB devices")?;
        let device = devices
            .iter()
            .find(|device| {
                device
                    .device_descriptor()
                    .is_ok_and(|d| d.vendor_id() == PS3_EYE_VID && d.product_id() == PS3_EYE_PID)
            })
            .context("PS3 Eye not found on the USB bus")?;

        let handle = device.open().context("failed to open the camera")?;

        // The reference driver picks the bulk IN endpoint of interface 0,
        // alternate setting 0.
        let config = device
            .active_config_descriptor()
            .context("failed to read the active config descriptor")?;
        let bulk_endpoint = config
            .interfaces()
            .find(|iface| iface.descriptors().next().is_some_and(|s| s.interface_number() == 0))
            .and_then(|iface| iface.descriptors().next())
            .and_then(|setting| {
                setting
                    .endpoint_descriptors()
                    .find(|ep| ep.transfer_type() == rusb::TransferType::Bulk)
            })
            .map(|ep| ep.address())
            .context("no bulk endpoint on interface 0")?;

        // libusb_claim_interface(handle_, 0)
        handle
            .claim_interface(0)
            .context("failed to claim interface 0")?;

        log::info!(
            "PS3 Eye opened: bus {:03} device {:03}, bulk endpoint {:#04x}",
            device.bus_number(),
            device.address(),
            bulk_endpoint
        );

        Ok(Self {
            bridge: Bridge::new(Arc::new(SharedHandle(handle))),
            bulk_endpoint,
            resolution: None,
            sensor_id: 0,
            fps: 0,
            streaming: false,
            stream: None,
        })
    }

    /// Sensor ID, as reported over SCCB (should be 0x7721 for the OV772x).
    #[must_use]
    pub const fn sensor_id(&self) -> u16 {
        self.sensor_id
    }

    #[must_use]
    pub const fn resolution(&self) -> Option<Resolution> {
        self.resolution
    }

    #[must_use]
    pub const fn fps(&self) -> u16 {
        self.fps
    }

    #[must_use]
    pub const fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Bring the bridge and sensor up and select resolution + frame rate.
    pub fn init(&mut self, resolution: Resolution, desired_fps: u16) -> Result<()> {
        let (width, _height) = resolution.dims();
        self.fps = ov534::set_frame_rate(&self.bridge, width, desired_fps, true)?;
        self.resolution = Some(resolution);

        /* reset bridge */
        self.bridge.reg_write(0xe7, 0x3a)?;
        self.bridge.reg_write(0xe0, 0x08)?;
        std::thread::sleep(Duration::from_millis(100));

        /* initialize the sensor address */
        self.bridge.reg_write(0xf1, 0x42)?;

        /* reset sensor */
        self.bridge.sccb_write(0x12, 0x80)?;
        std::thread::sleep(Duration::from_millis(10));

        /* probe the sensor */
        let _ = self.bridge.sccb_read(0x0a)?;
        self.sensor_id = u16::from(self.bridge.sccb_read(0x0a)?) << 8;
        let _ = self.bridge.sccb_read(0x0b)?;
        self.sensor_id |= u16::from(self.bridge.sccb_read(0x0b)?);
        log::info!("Sensor ID: {:#06x}", self.sensor_id);

        /* initialize */
        self.bridge.write_bridge_table(ov534::BRIDGE_INIT)?;
        self.bridge.set_led(true)?;
        self.bridge.write_sensor_table(ov534::SENSOR_INIT)?;
        self.bridge.reg_write(0xe0, 0x09)?;
        self.bridge.set_led(false)?;

        Ok(())
    }

    /// Start streaming.
    pub fn start(&mut self) -> Result<()> {
        if self.streaming {
            return Ok(());
        }
        let resolution = self
            .resolution
            .context("init() must be called before start()")?;

        let (bridge_start, sensor_start) = match resolution {
            Resolution::Qvga => (ov534::BRIDGE_START_QVGA, ov534::SENSOR_START_QVGA),
            Resolution::Vga => (ov534::BRIDGE_START_VGA, ov534::SENSOR_START_VGA),
        };
        self.bridge.write_bridge_table(bridge_start)?;
        self.bridge.write_sensor_table(sensor_start)?;

        self.fps = ov534::set_frame_rate(&self.bridge, resolution.dims().0, self.fps, false)?;

        self.bridge.set_led(true)?;
        self.bridge.reg_write(0xe0, 0x00)?; // start the stream

        self.bridge
            .handle()
            .clear_halt(self.bulk_endpoint)
            .context("clear_halt on the bulk endpoint failed")?;

        let scanner = PktScanner::new(resolution.frame_size());
        self.stream = Some(AsyncStream::start(
            self.bridge.handle(),
            self.bulk_endpoint,
            scanner,
        )?);

        self.streaming = true;
        Ok(())
    }

    /// Stop streaming.
    pub fn stop(&mut self) -> Result<()> {
        if !self.streaming {
            return Ok(());
        }
        /* stop streaming data */
        self.bridge.reg_write(0xe0, 0x09)?;
        self.bridge.set_led(false)?;

        // Dropping the stream cancels its transfers and joins its event thread.
        self.stream = None;
        self.streaming = false;
        Ok(())
    }

    /// Block until the next complete raw Bayer frame arrives.
    ///
    /// The returned vector is exactly `width * height` bytes.
    pub fn read_raw_frame(&self) -> Result<Vec<u8>> {
        ensure!(self.streaming, "streaming is not active");
        let stream = self.stream.as_ref().context("stream is missing")?;
        stream.next_frame(FRAME_TIMEOUT)
    }

    /// Frames dropped because a newer one was already queued (latency guard).
    pub fn skipped_frames(&self) -> usize {
        self.stream.as_ref().map_or(0, AsyncStream::skipped)
    }
}

impl Drop for Ps3Eye {
    fn drop(&mut self) {
        if let Err(err) = self.stop() {
            log::warn!("error while stopping the stream on drop: {err:#}");
        }
    }
}
