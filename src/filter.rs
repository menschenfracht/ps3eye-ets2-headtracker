//! One-Euro filter for head-pose smoothing.
//!
//! Adaptive low-pass: it smooths when the signal is slow and becomes
//! responsive when it moves fast. See Casiez, Roussel & Vogel, "1€ Filter"
//! (CHI 2012).

use std::f32::consts::PI;

/// A one-dimensional One-Euro filter.
#[derive(Clone, Copy, Debug)]
pub struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    value: f32,
    derivative: f32,
    initialized: bool,
}

impl OneEuro {
    #[must_use]
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            value: 0.0,
            derivative: 0.0,
            initialized: false,
        }
    }

    /// Filter `value` given the time step `dt` in seconds.
    pub fn filter(&mut self, value: f32, dt: f32) -> f32 {
        if !self.initialized || dt <= 0.0 {
            self.value = value;
            self.derivative = 0.0;
            self.initialized = true;
            return value;
        }

        let derivative = (value - self.value) / dt;
        let smoothed_derivative = lowpass(derivative, self.derivative, alpha(dt, self.d_cutoff));
        let cutoff = self.min_cutoff + self.beta * smoothed_derivative.abs();
        let filtered = lowpass(value, self.value, alpha(dt, cutoff));

        self.value = filtered;
        self.derivative = smoothed_derivative;
        filtered
    }
}

/// Filter coefficient for a cutoff frequency and time step.
fn alpha(dt: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

fn lowpass(value: f32, previous: f32, alpha: f32) -> f32 {
    previous + alpha * (value - previous)
}
