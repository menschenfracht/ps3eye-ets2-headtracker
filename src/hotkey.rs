//! Global recenter hotkey (Ctrl+Alt+R) via a low-level keyboard hook.
//!
//! `RegisterHotKey` only removes the keystroke from the normal message queue:
//! games that read the keyboard through raw input or DirectInput (ETS2 does)
//! still receive the key and act on it — our recenter also opened ETS2's radio
//! menu. A `WH_KEYBOARD_LL` hook runs before the input reaches any application,
//! so returning `1` swallows the keystroke for everyone.
//!
//! The hook lives on a dedicated thread that owns a message loop, as Windows
//! requires, and only swallows `R` while Ctrl and Alt are held.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;

use anyhow::{Context, Result};

/// Virtual-key code for the `R` key.
const VK_R: u32 = 0x52;
/// Virtual-key codes for the modifiers that must be held.
const VK_CONTROL: i32 = 0x11;
const VK_MENU: i32 = 0x12; // Alt

/// `WH_KEYBOARD_LL`.
const WH_KEYBOARD_LL: i32 = 13;
/// `WM_KEYDOWN` / `WM_SYSKEYDOWN` / `WM_KEYUP` / `WM_SYSKEYUP`.
const WM_KEYDOWN: usize = 0x0100;
const WM_SYSKEYDOWN: usize = 0x0104;
const WM_KEYUP: usize = 0x0101;
const WM_SYSKEYUP: usize = 0x0105;

/// Caller's flag, set on each hotkey press.
static PRESSED: OnceLock<Arc<AtomicBool>> = OnceLock::new();
/// Whether `R` is currently held (so auto-repeat fires once).
static DOWN: AtomicBool = AtomicBool::new(false);
/// Whether the matching key-up still has to be swallowed.
static SWALLOW_UP: AtomicBool = AtomicBool::new(false);

/// Spawn a thread that watches for Ctrl+Alt+R and sets `pressed` on each hit.
///
/// Returns once the hook is installed (or with an error).
#[cfg(windows)]
pub fn spawn(pressed: Arc<AtomicBool>) -> Result<()> {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
    let _ = PRESSED.set(pressed);

    thread::Builder::new()
        .name("recenter-hotkey".to_owned())
        .spawn(move || {
            // SAFETY: Win32 calls have no memory-safety requirements; this
            // thread owns the message loop the hook is dispatched on.
            unsafe {
                let module = GetModuleHandleW(std::ptr::null());
                let hook = SetWindowsHookExW(
                    WH_KEYBOARD_LL,
                    Some(keyboard_proc),
                    module,
                    0,
                );
                if hook.is_null() {
                    let _ = ready_tx.send(Err(anyhow::anyhow!(
                        "SetWindowsHookExW(WH_KEYBOARD_LL) failed"
                    )));
                    return;
                }
                let _ = ready_tx.send(Ok(()));

                // A low-level hook is called on the installing thread, which
                // must keep pumping messages.
                let mut message = Msg::default();
                while GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) > 0 {}
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

/// Hook procedure: fire on Ctrl+Alt+R and swallow the keystroke.
#[cfg(windows)]
unsafe extern "system" fn keyboard_proc(code: i32, wparam: usize, lparam: isize) -> isize {
    // SAFETY: for `code >= 0` Windows passes a valid KBDLLHOOKSTRUCT.
    unsafe {
        if code < 0 {
            return CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam);
        }
        // SAFETY: as above.
        let event = &*(lparam as *const KbdLlHookStruct);
        if event.vk_code == VK_R {
            let is_down = wparam == WM_KEYDOWN || wparam == WM_SYSKEYDOWN;
            let is_up = wparam == WM_KEYUP || wparam == WM_SYSKEYUP;
            if is_down && modifiers_held() {
                if !DOWN.swap(true, Ordering::SeqCst) {
                    if let Some(pressed) = PRESSED.get() {
                        pressed.store(true, Ordering::SeqCst);
                    }
                }
                SWALLOW_UP.store(true, Ordering::SeqCst);
                return 1;
            }
            if is_up {
                let swallowed = SWALLOW_UP.swap(false, Ordering::SeqCst);
                DOWN.store(false, Ordering::SeqCst);
                if swallowed {
                    return 1;
                }
            }
        }
        CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam)
    }
}

/// Whether Ctrl and Alt are both down right now.
#[cfg(windows)]
fn modifiers_held() -> bool {
    // SAFETY: no memory-safety requirements.
    let ctrl = unsafe { GetAsyncKeyState(VK_CONTROL) } as u16 & 0x8000 != 0;
    let alt = unsafe { GetAsyncKeyState(VK_MENU) } as u16 & 0x8000 != 0;
    ctrl && alt
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

/// Win32 `KBDLLHOOKSTRUCT`.
#[cfg(windows)]
#[repr(C)]
struct KbdLlHookStruct {
    vk_code: u32,
    scan_code: u32,
    flags: u32,
    time: u32,
    dw_extra_info: usize,
}

#[cfg(windows)]
unsafe extern "system" {
    fn GetMessageW(msg: *mut Msg, hwnd: *mut core::ffi::c_void, min: u32, max: u32) -> i32;
    fn SetWindowsHookExW(
        id_hook: i32,
        callback: Option<unsafe extern "system" fn(i32, usize, isize) -> isize>,
        module: *mut core::ffi::c_void,
        thread_id: u32,
    ) -> *mut core::ffi::c_void;
    fn CallNextHookEx(
        hook: *mut core::ffi::c_void,
        code: i32,
        wparam: usize,
        lparam: isize,
    ) -> isize;
    fn GetAsyncKeyState(vk: i32) -> i16;
    fn GetModuleHandleW(module_name: *const u16) -> *mut core::ffi::c_void;
}
