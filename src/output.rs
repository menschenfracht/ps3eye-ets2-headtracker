//! Send head pose to opentrack over UDP, using its FreePIE receiver format.
//!
//! opentrack's "FreePIE UDP receiver" input expects a 50-byte packet:
//! one pad byte, one flags byte, then twelve little-endian `f32`. With
//! `flags = 2` (orientation) the first three floats are yaw, pitch and roll
//! in radians. The default port is 5555. opentrack then presents the pose as
//! a TrackIR device to the game.

use std::net::{SocketAddr, UdpSocket};

use anyhow::{Context, Result};

use crate::pose::Pose;

/// FreePIE packet flag selecting orientation-only data.
const FLAG_ORIENT: u8 = 2;
/// Default opentrack FreePIE receiver port.
pub const DEFAULT_PORT: u16 = 5555;

pub struct OpenTrack {
    socket: UdpSocket,
    addr: SocketAddr,
}

impl OpenTrack {
    /// Open a local UDP socket that sends to `127.0.0.1:port`.
    pub fn connect(port: u16) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0").context("failed to bind UDP socket")?;
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        Ok(Self { socket, addr })
    }

    /// Send a pose in radians.
    pub fn send(&self, pose: &Pose) -> Result<()> {
        let mut packet = Vec::with_capacity(50);
        packet.push(0); // pad
        packet.push(FLAG_ORIENT);
        let floats = [
            pose.yaw.to_radians(),
            pose.pitch.to_radians(),
            pose.roll.to_radians(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for value in floats {
            packet.extend_from_slice(&value.to_le_bytes());
        }
        self.socket
            .send_to(&packet, self.addr)
            .context("failed to send pose to opentrack")?;
        Ok(())
    }
}
