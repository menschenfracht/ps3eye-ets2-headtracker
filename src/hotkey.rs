//! Global recenter hotkey (Ctrl+Alt+R) via Win32 `RegisterHotKey`.
//!
//! The hotkey is registered on a dedicated thread that owns a Win32 message
//! loop, so Windows delivers `WM_HOTKEY` to us even while another application
//! (ETS2) has focus.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result};

/// Virtual-key code for the `R` key.
const VK_R: u32 = 0x52;
/// `MOD_ALT | MOD_CONTROL | MOD_NOREPEAT`.
const MOD_ALT_CTRL: u32 = 0x0001 | 0x0002 | 0x4000;
/// Arbitrary hotkey id.
const HOTKEY_ID: i32 = 1;
/// `WM_HOTKEY`.
const WM_HOTKEY: u32 = 0x0312;

/// Spawn a thread that watches for Ctrl+Alt+R and sets `pressed` on each hit.
///
/// Returns once the hotkey is registered (or with an error if it could not
/// be, e.g. another application already owns it).
#[cfg(windows)]
pub fn spawn(pressed: Arc<AtomicBool>) -> Result<()> {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
    thread::Builder::new()
        .name("recenter-hotkey".to_owned())
        .spawn(move || {
            // SAFETY: Win32 calls have no memory-safety requirements; this
            // thread owns the message queue the hotkey posts to.
            let registered =
                unsafe { RegisterHotKey(std::ptr::null_mut(), HOTKEY_ID, MOD_ALT_CTRL, VK_R) };
            if registered == 0 {
                let _ = ready_tx.send(Err(anyhow::anyhow!(
                    "RegisterHotKey(Ctrl+Alt+R) failed (already in use?)"
                )));
                return;
            }
            let _ = ready_tx.send(Ok(()));

            let mut message = Msg::default();
            loop {
                // SAFETY: `message` is a valid MSG for the duration of the call.
                let ret = unsafe { GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) };
                if ret == 0 || ret == -1 {
                    break;
                }
                if message.message == WM_HOTKEY {
                    pressed.store(true, Ordering::SeqCst);
                }
            }
        })
        .context("failed to spawn the hotkey thread")?;
    ready_rx.recv().context("hotkey thread did not report ready")?
}

/// No-op outside Windows.
#[cfg(not(windows))]
pub fn spawn(_pressed: Arc<AtomicBool>) -> Result<()> {
    Ok(())
}

/// Win32 `MSG`.
#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct Msg {
    hwnd: *mut core::ffi::c_void,
    message: u32,
    w_param: usize,
    l_param: isize,
    time: u32,
    pt_x: i32,
    pt_y: i32,
}

#[cfg(windows)]
unsafe extern "system" {
    fn RegisterHotKey(
        hwnd: *mut core::ffi::c_void,
        id: i32,
        fs_modifiers: u32,
        vk: u32,
    ) -> i32;
    fn GetMessageW(msg: *mut Msg, hwnd: *mut core::ffi::c_void, min: u32, max: u32) -> i32;
}
