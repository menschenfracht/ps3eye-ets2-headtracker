//! Raw payload scanner: assembles raw Bayer frames out of bulk UVC payloads.
//! (Port of gspca's `pkt_scan`/`frame_add` via PS3EYEDriver.)

use std::collections::VecDeque;

/// Bulk payload unit size used by the camera.
pub const PAYLOAD_LEN: usize = 2_048;

/// UVC-style payload header flags (subset used by the scanner).
const STREAM_EOF: u8 = 1 << 1;
const STREAM_PTS: u8 = 1 << 2;
const STREAM_FID: u8 = 1 << 0;
const STREAM_ERR: u8 = 1 << 6;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PacketType {
    Discard,
    First,
    Inter,
    Last,
}

/// Assembles raw Bayer frames out of bulk UVC payloads.
pub struct PktScanner {
    frame_size: usize,
    last_pts: u32,
    last_fid: u16,
    last_packet_type: PacketType,
    cur_frame: Vec<u8>,
    completed: VecDeque<Vec<u8>>,
}

impl PktScanner {
    pub fn new(frame_size: usize) -> Self {
        Self {
            frame_size,
            last_pts: 0,
            last_fid: 0,
            last_packet_type: PacketType::Discard,
            cur_frame: Vec::with_capacity(frame_size),
            completed: VecDeque::new(),
        }
    }

    /// Feed one bulk transfer worth of data through the scanner.
    pub fn feed(&mut self, data: &[u8]) {
        let mut remaining = data;
        while !remaining.is_empty() {
            let len = remaining.len().min(PAYLOAD_LEN);
            let (payload, rest) = remaining.split_at(len);
            self.scan_payload(payload);
            remaining = rest;
        }
    }

    /// Take the oldest completed frame, if any.
    pub fn take_frame(&mut self) -> Option<Vec<u8>> {
        self.completed.pop_front()
    }

    /// Port of gspca's `frame_add`.
    fn frame_add(&mut self, mut packet_type: PacketType, data: &[u8]) {
        if packet_type == PacketType::First {
            self.cur_frame.clear();
        } else {
            match self.last_packet_type {
                PacketType::Discard => {
                    if packet_type == PacketType::Last {
                        self.last_packet_type = PacketType::Last;
                        self.cur_frame.clear();
                    }
                    return;
                }
                PacketType::Last => return,
                PacketType::First | PacketType::Inter => {}
            }
        }

        if !data.is_empty() {
            if self.cur_frame.len() + data.len() > self.frame_size {
                packet_type = PacketType::Discard;
                self.cur_frame.clear();
            } else {
                self.cur_frame.extend_from_slice(data);
            }
        }

        self.last_packet_type = packet_type;

        if packet_type == PacketType::Last {
            self.completed.push_back(std::mem::take(&mut self.cur_frame));
        }
    }

    /// Port of gspca's `pkt_scan` for the bulk case.
    fn scan_payload(&mut self, payload: &[u8]) {
        if payload.len() < 12 || payload[0] != 12 {
            log::debug!("bad header");
            self.frame_add(PacketType::Discard, &[]);
            return;
        }

        let flags = payload[1];
        if flags & STREAM_ERR != 0 {
            log::debug!("payload error");
            self.frame_add(PacketType::Discard, &[]);
            return;
        }
        if flags & STREAM_PTS == 0 {
            log::debug!("PTS not present");
            self.frame_add(PacketType::Discard, &[]);
            return;
        }

        let this_pts = u32::from(payload[2])
            | u32::from(payload[3]) << 8
            | u32::from(payload[4]) << 16
            | u32::from(payload[5]) << 24;
        let this_fid = u16::from(flags & STREAM_FID);

        if this_pts != self.last_pts || this_fid != self.last_fid {
            if self.last_packet_type == PacketType::Inter {
                // The last frame was incomplete; don't keep it.
                self.frame_add(PacketType::Discard, &[]);
            }
            self.last_pts = this_pts;
            self.last_fid = this_fid;
            self.frame_add(PacketType::First, &payload[12..]);
        } else if flags & STREAM_EOF != 0 {
            self.last_pts = 0;
            if self.cur_frame.len() + payload.len() - 12 != self.frame_size {
                log::debug!("frame length mismatch at EOF");
                self.frame_add(PacketType::Discard, &[]);
            } else {
                self.frame_add(PacketType::Last, &payload[12..]);
            }
        } else {
            self.frame_add(PacketType::Inter, &payload[12..]);
        }
    }
}


