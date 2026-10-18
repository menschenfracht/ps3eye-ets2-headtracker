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

    /// Send a pose (degrees) to opentrack.
    pub fn send(&self, pose: &Pose) -> Result<()> {
        self.socket
            .send_to(&build_packet(pose), self.addr)
            .context("failed to send pose to opentrack")?;
        Ok(())
    }
}

/// Build the 50-byte FreePIE orientation packet: pad, flags, 12 LE `f32`.
fn build_packet(pose: &Pose) -> Vec<u8> {
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
    packet
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_packet_layout() {
        let pose = Pose {
            yaw: 90.0,
            pitch: -45.0,
            roll: 10.0,
        };
        let packet = build_packet(&pose);

        assert_eq!(packet.len(), 50);
        assert_eq!(packet.first(), Some(&0));
        assert_eq!(packet.get(1), Some(&FLAG_ORIENT));

        let read = |offset: usize| {
            f32::from_le_bytes(
                packet
                    .get(offset..offset + 4)
                    .and_then(|b| b.try_into().ok())
                    .expect("float slot inside the packet"),
            )
        };
        assert!((read(2) - 90.0_f32.to_radians()).abs() < 1.0e-6);
        assert!((read(6) - (-45.0_f32).to_radians()).abs() < 1.0e-6);
        assert!((read(10) - 10.0_f32.to_radians()).abs() < 1.0e-6);
    }
}
