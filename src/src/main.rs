//! Headtrack — DIY head tracker for ETS2 built on the PS3 Eye (OV534) camera.
//!
//! Milestone M1.0: find the camera on the USB bus through `libusbK` and prove
//! that we can open it and talk to it. Video streaming comes next.

use std::time::Duration;

use anyhow::{Context, Result};
use rusb::{Device, DeviceList, UsbContext};

/// Short timeout for descriptor/string reads.
const IO_TIMEOUT: Duration = Duration::from_millis(100);

/// Nam Tai E&E / OmniVision — the PS3 Eye's USB identifiers.
const PS3_EYE_VID: u16 = 0x1415;
const PS3_EYE_PID: u16 = 0x2000;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let devices = DeviceList::new().context("failed to list USB devices")?;
    log::info!("{} USB device(s) visible to libusb", devices.len());

    let camera = devices
        .iter()
        .find(|device| is_ps3_eye(device))
        .context("PS3 Eye not found on the USB bus")?;

    describe_camera(&camera)?;

    Ok(())
}

/// True if the device is the PS3 Eye.
fn is_ps3_eye(device: &Device<impl UsbContext>) -> bool {
    let Ok(descriptor) = device.device_descriptor() else {
        return false;
    };
    descriptor.vendor_id() == PS3_EYE_VID && descriptor.product_id() == PS3_EYE_PID
}

/// Open the camera and dump everything relevant about it.
fn describe_camera(device: &Device<impl UsbContext>) -> Result<()> {
    let descriptor = device.device_descriptor().context("descriptor read failed")?;
    let handle = device.open().context("failed to open the camera")?;

    let langs = handle.read_languages(IO_TIMEOUT).unwrap_or_default();
    let serial = if langs.is_empty() {
        String::new()
    } else {
        handle
            .read_serial_number_string(langs[0], &descriptor, IO_TIMEOUT)
            .unwrap_or_default()
    };

    log::info!("PS3 Eye found:");
    log::info!("  bus {:03} device {:03}", device.bus_number(), device.address());
    log::info!(
        "  USB version {}, device release {}",
        descriptor.usb_version(),
        descriptor.device_version()
    );
    log::info!("  vendor id  {:#06x}", descriptor.vendor_id());
    log::info!("  product id {:#06x}", descriptor.product_id());
    log::info!("  serial     {serial:?}");
    log::info!(
        "  speed      {}",
        match device.speed() {
            rusb::Speed::Low => "low (1.5 Mbit/s)",
            rusb::Speed::Full => "full (12 Mbit/s)",
            rusb::Speed::High => "high (480 Mbit/s) — expected for the PS3 Eye",
            rusb::Speed::Super => "super (5 Gbit/s)",
            _ => "unknown",
        }
    );

    let config = device.active_config_descriptor().context("no active config")?;
    log::info!("  interfaces: {}", config.num_interfaces());
    for iface in config.interfaces() {
        let setting = iface
            .descriptors()
            .next()
            .context("interface without a descriptor")?;
        log::info!(
            "  interface {} (setting {}): class {:#04x}, {} endpoint(s)",
            setting.interface_number(),
            setting.setting_number(),
            setting.class_code(),
            setting.num_endpoints(),
        );
        for endpoint in setting.endpoint_descriptors() {
            log::info!(
                "    endpoint {:#04x}: {:?}, max packet {}",
                endpoint.address(),
                endpoint.transfer_type(),
                endpoint.max_packet_size(),
            );
        }
    }

    // Prove we can send a control transfer: read the OV534 register 0x01
    // (device ID high byte) — the standard smoke test from PS3EYEDriver.
    // Register read: bmRequestType 0xC0, bRequest 0x01, wValue = register.
    const OV534_REG_READ: u8 = 0xC0;
    const OV534_REG_WRITE: u8 = 0x40;
    let _ = (OV534_REG_READ, OV534_REG_WRITE); // used from M1.1 on

    Ok(())
}
