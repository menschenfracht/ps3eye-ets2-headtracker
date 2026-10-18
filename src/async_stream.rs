//! Asynchronous bulk URB ring for the PS3 Eye stream.
//!
//! Mirrors the reference driver (`PS3EYEDriver`, `URBDesc`): a fixed set of
//! libusb bulk transfers stays in flight, and a single event thread runs the
//! completion callbacks. Each completed transfer is scanned and immediately
//! resubmitted, so chunks are processed in the order the bus delivers them.
//! (Our earlier design read from several threads and handed chunks over a
//! channel; those threads could hand chunks over out of order.)

use std::cell::UnsafeCell;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use libc::{c_int, c_void, timeval};
use libusb1_sys as ffi;
use rusb::{DeviceHandle, GlobalContext};

use crate::scanner::PktScanner;

/// Transfers kept in flight (the reference driver uses five).
const TRANSFERS: usize = 5;
/// Size of each bulk transfer.
const TRANSFER_SIZE: usize = 65_536;
/// How long the event loop waits before re-checking the stop condition.
const EVENT_TICK: timeval = timeval {
    tv_sec: 0,
    tv_usec: 50_000,
};

/// libusb `LIBUSB_TRANSFER_COMPLETED` / `LIBUSB_TRANSFER_CANCELLED`.
const TRANSFER_COMPLETED: c_int = 0;
const TRANSFER_CANCELLED: c_int = 3;

/// State shared between the owner and the libusb callbacks.
struct Ctx {
    /// Only touched from callbacks, which all run on the event thread.
    scanner: UnsafeCell<PktScanner>,
    frame_tx: mpsc::Sender<Vec<u8>>,
    /// Transfers currently owned by libusb (submitted, not yet completed).
    pending: AtomicUsize,
    stopping: AtomicBool,
}

/// Raw context pointer that may cross into the event thread.
struct CtxPtr(*mut Ctx);

// SAFETY: the pointee is only accessed through the atomics, or from the
// event thread for the scanner; it outlives the thread (see `Drop`).
unsafe impl Send for CtxPtr {}

impl CtxPtr {
    /// Accessor, so closures capture the whole wrapper (and its `Send` impl).
    fn get(&self) -> *mut Ctx {
        self.0
    }
}

/// A running asynchronous bulk stream delivering complete raw frames.
pub struct AsyncStream {
    ctx: *mut Ctx,
    transfers: Vec<*mut ffi::libusb_transfer>,
    /// Owns the transfer buffers; the transfers point into these.
    buffers: Vec<Vec<u8>>,
    event_thread: Option<JoinHandle<()>>,
    frame_rx: mpsc::Receiver<Vec<u8>>,
}

impl AsyncStream {
    /// Submit the transfer ring on `endpoint` and start the event thread.
    pub fn start(
        handle: &DeviceHandle<GlobalContext>,
        endpoint: u8,
        scanner: PktScanner,
    ) -> Result<Self> {
        let (frame_tx, frame_rx) = mpsc::channel();
        let ctx = Box::into_raw(Box::new(Ctx {
            scanner: UnsafeCell::new(scanner),
            frame_tx,
            pending: AtomicUsize::new(0),
            stopping: AtomicBool::new(false),
        }));

        let mut stream = Self {
            ctx,
            transfers: Vec::with_capacity(TRANSFERS),
            buffers: (0..TRANSFERS).map(|_| vec![0_u8; TRANSFER_SIZE]).collect(),
            event_thread: None,
            frame_rx,
        };

        let ctx_ptr = CtxPtr(ctx);
        stream.event_thread = Some(std::thread::spawn(move || {
            let ctx = ctx_ptr.get();
            loop {
                // SAFETY: `ctx` is alive until this thread is joined in `Drop`.
                let state = unsafe { &*ctx };
                if state.stopping.load(Ordering::SeqCst) && state.pending.load(Ordering::SeqCst) == 0 {
                    break;
                }
                let tick = EVENT_TICK;
                // SAFETY: a null context selects libusb's default context,
                // which is the one rusb opened the device on.
                unsafe {
                    ffi::libusb_handle_events_timeout_completed(ptr::null_mut(), &tick, ptr::null_mut());
                }
            }
        }));

        let dev = handle.as_raw();
        for index in 0..TRANSFERS {
            // SAFETY: allocation and fill of a bulk transfer on a valid handle.
            let xfr = unsafe { ffi::libusb_alloc_transfer(0) };
            if xfr.is_null() {
                bail!("libusb_alloc_transfer failed");
            }
            stream.transfers.push(xfr);
            let buffer = stream.buffers[index].as_mut_ptr();
            // SAFETY: `buffer` lives as long as `stream`, which outlives the transfer.
            unsafe {
                ffi::libusb_fill_bulk_transfer(
                    xfr,
                    dev,
                    endpoint,
                    buffer,
                    c_int::try_from(TRANSFER_SIZE).expect("transfer size fits c_int"),
                    on_transfer,
                    ctx.cast::<c_void>(),
                    0,
                );
                (*ctx).pending.fetch_add(1, Ordering::SeqCst);
                let rc = ffi::libusb_submit_transfer(xfr);
                if rc != 0 {
                    (*ctx).pending.fetch_sub(1, Ordering::SeqCst);
                    bail!("libusb_submit_transfer failed with code {rc}");
                }
            }
        }

        Ok(stream)
    }

    /// Wait for the next complete frame.
    pub fn next_frame(&self, timeout: Duration) -> Result<Vec<u8>> {
        self.frame_rx
            .recv_timeout(timeout)
            .context("no frame from the camera (stream stalled or stopped)")
    }
}

/// Completion callback; runs on the event thread.
extern "system" fn on_transfer(xfr: *mut ffi::libusb_transfer) {
    // SAFETY: `user_data` is the `Ctx` pointer given in `start`; libusb only
    // calls this on the event thread, which `Drop` joins before freeing `Ctx`.
    unsafe {
        let ctx = &*((*xfr).user_data.cast::<Ctx>());
        ctx.pending.fetch_sub(1, Ordering::SeqCst);

        match (*xfr).status {
            TRANSFER_COMPLETED => {
                let len = usize::try_from((*xfr).actual_length).unwrap_or(0);
                let data = std::slice::from_raw_parts((*xfr).buffer, len);
                let scanner = &mut *ctx.scanner.get();
                scanner.feed(data);
                while let Some(frame) = scanner.take_frame() {
                    // The receiver may be gone during shutdown; that's fine.
                    let _ = ctx.frame_tx.send(frame);
                }
            }
            TRANSFER_CANCELLED => {}
            other => {
                log::warn!("bulk transfer ended with status {other}; not resubmitting");
                return;
            }
        }

        if !ctx.stopping.load(Ordering::SeqCst) {
            ctx.pending.fetch_add(1, Ordering::SeqCst);
            if ffi::libusb_submit_transfer(xfr) != 0 {
                ctx.pending.fetch_sub(1, Ordering::SeqCst);
                log::warn!("resubmitting bulk transfer failed");
            }
        }
    }
}

impl Drop for AsyncStream {
    fn drop(&mut self) {
        // SAFETY: `ctx` was created in `start` and is freed only below, after
        // the event thread (the only user of the callbacks) has finished.
        unsafe {
            (*self.ctx).stopping.store(true, Ordering::SeqCst);
            for &xfr in &self.transfers {
                ffi::libusb_cancel_transfer(xfr);
            }
        }
        if let Some(thread) = self.event_thread.take() {
            if thread.join().is_err() {
                log::warn!("event thread panicked during shutdown");
            }
        }
        // SAFETY: no callbacks can run anymore; transfers are idle.
        unsafe {
            for &xfr in &self.transfers {
                ffi::libusb_free_transfer(xfr);
            }
            drop(Box::from_raw(self.ctx));
        }
    }
}
