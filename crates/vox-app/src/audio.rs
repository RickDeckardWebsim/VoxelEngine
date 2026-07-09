//! SM64 audio playback: a cpal output stream fed by a background thread
//! that calls `sm64_audio_tick` to synthesize sound effects.
//!
//! libsm64's audio runs at 32000 Hz, stereo, signed 16-bit PCM, interleaved
//! L/R. `sm64_audio_tick` is *not* driven by the render loop — it maintains
//! its own internal sequencing and is designed to be called from a separate
//! thread (matching the upstream SDL example, which runs audio on its own
//! pthread while the main loop ticks Mario). We replicate that split:
//!
//!   - a **feeder thread** calls `sm64_audio_tick` ~30×/s and pushes the
//!     produced samples into a lock-free SPSC ring buffer;
//!   - a **cpal output stream** pulls from that ring in its real-time
//!     callback, writing silence when the buffer underruns.
//!
//! Back-pressure mirrors the SDL example: `sm64_audio_tick` is told how many
//! samples are still queued so it can shrink its output, and the feeder only
//! pushes when there is room — preventing unbounded growth.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, StreamConfig};

use vox_sm64::ffi;

/// SM64 audio sample rate (Hz).
const SM64_SAMPLE_RATE: u32 = 32_000;
/// SM64 audio channel count (stereo, interleaved L/R).
const SM64_CHANNELS: u16 = 2;
/// `numDesiredSamples` passed to `sm64_audio_tick` (from the upstream
/// SDL example). When fewer than this many samples are queued, SM64
/// produces `SAMPLES_HIGH` (544) per channel; otherwise `SAMPLES_LOW`
/// (528). Either way the buffer must hold `544 * 2 * 2` i16 values.
const NUM_DESIRED_SAMPLES: u32 = 1100;
/// Max i16 values `sm64_audio_tick` can write per call: 2 channels ×
/// 544 per-channel samples × 2 (the `2 * num_audio_samples` stride in
/// `create_next_audio_buffer`). Matches the SDL example's buffer.
const TICK_BUFFER_LEN: usize = 544 * 2 * 2;
/// Ring-buffer capacity in i16 samples (power of two → maskable indices).
/// 32768 samples = 16384 stereo frames ≈ 0.5 s at 32 kHz — comfortably
/// above the SDL example's 6000-frame queue ceiling, absorbing feeder
/// jitter without overflowing.
const RING_CAPACITY: usize = 32_768;
const RING_MASK: usize = RING_CAPACITY - 1;

/// Target feeder cadence: ~30 Hz, matching SM64's internal tick rate and
/// the upstream SDL example's 33 ms loop.
const FEEDER_SLEEP: Duration = Duration::from_millis(33);

// ── SPSC ring buffer ───────────────────────────────────────────────────

/// A single-producer / single-consumer ring buffer of `i16` samples.
///
/// The feeder thread is the sole producer; the cpal callback is the sole
/// consumer. Indices advance with `Relaxed` atomics — correct because there
/// is exactly one reader and one writer, each tracking only its own index
/// and observing the other's via the matching `Acquire`/`Release` pair.
struct RingBuffer {
    /// Backing storage; `UnsafeCell` because the single producer and
    /// single consumer touch disjoint regions (enforced by the index
    /// checks below) but both hold `&RingBuffer` via the shared `Arc`.
    buf: std::cell::UnsafeCell<Box<[i16]>>,
    /// Total slots written by the producer (monotonic; masked on use).
    write: AtomicUsize,
    /// Total slots read by the consumer (monotonic; masked on use).
    read: AtomicUsize,
}

// SAFETY: The ring is shared between exactly one producer (feeder thread)
// and one consumer (cpal callback). They never access the same slot: the
// producer only writes within `[write, write+space)` and the consumer only
// reads within `[read, read+avail)`, and `space` is computed so those
// ranges never overlap. `Arc<RingBuffer>` is therefore safe to share.
unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    fn new() -> Self {
        let buf = vec![0i16; RING_CAPACITY].into_boxed_slice();
        Self {
            buf: std::cell::UnsafeCell::new(buf),
            write: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
        }
    }

    /// Free slots available for the producer to write.
    fn write_space(&self) -> usize {
        let w = self.write.load(Ordering::Relaxed);
        let r = self.read.load(Ordering::Acquire);
        RING_CAPACITY - (w.wrapping_sub(r))
    }

    /// Number of samples the consumer can read right now.
    fn read_space(&self) -> usize {
        let w = self.write.load(Ordering::Acquire);
        let r = self.read.load(Ordering::Relaxed);
        w.wrapping_sub(r)
    }

    /// Push `samples` into the ring. Returns the number actually pushed;
    /// any that don't fit are dropped (back-pressure: the feeder drops
    /// overflow rather than blocking the audio subsystem).
    fn push(&self, samples: &[i16]) -> usize {
        let space = self.write_space();
        let n = samples.len().min(space);
        if n == 0 {
            return 0;
        }
        let w = self.write.load(Ordering::Relaxed);
        // SAFETY: sole producer; slots [w, w+n) are free (verified by
        // write_space) and disjoint from the consumer's read window.
        let buf = unsafe { &mut *self.buf.get() };
        for (i, &s) in samples[..n].iter().enumerate() {
            buf[(w + i) & RING_MASK] = s;
        }
        // Writes visible to consumer before advancing the write cursor.
        self.write.store(w.wrapping_add(n), Ordering::Release);
        n
    }

    /// Pop up to `out.len()` samples into `out`. Returns the count filled;
    /// the remainder is left untouched (the cpal callback fills gaps with
    /// silence so underruns click rather than crash).
    fn pop(&self, out: &mut [i16]) -> usize {
        let avail = self.read_space();
        let n = out.len().min(avail);
        if n == 0 {
            return 0;
        }
        let r = self.read.load(Ordering::Relaxed);
        // SAFETY: sole consumer; slots [r, r+n) are populated (verified by
        // read_space) and disjoint from the producer's write window.
        let buf = unsafe { &*self.buf.get() };
        for (i, slot) in out.iter_mut().take(n).enumerate() {
            *slot = buf[(r + i) & RING_MASK];
        }
        // Reads complete before advancing the read cursor.
        self.read.store(r.wrapping_add(n), Ordering::Release);
        n
    }
}

// ── Public handle ──────────────────────────────────────────────────────

/// Owns the SM64 audio subsystem: a cpal output stream plus the feeder
/// thread that keeps it supplied. Drop stops both.
pub struct Sm64Audio {
    ring: Arc<RingBuffer>,
    stream: Option<cpal::Stream>,
    feeder: Option<thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl Sm64Audio {
    /// Initialize libsm64's audio banks from the ROM. Must be called once
    /// before [`Sm64Audio::start`]. Safe to call even if audio playback
    /// later fails to open a device — the C side tolerates tick calls
    /// after a failed init by returning 0.
    pub fn init(rom: &[u8]) {
        // SAFETY: `sm64_audio_init` reads the ROM bytes and stashes them in
        // libsm64's global audio state; it does not retain the pointer.
        // We pass a valid, NUL-bounded slice (the ROM is a fixed binary).
        unsafe {
            ffi::sm64_audio_init(rom.as_ptr());
        }
        tracing::info!("SM64 audio initialized (banks loaded from ROM)");
    }

    /// Open the default output device at 32 kHz stereo and spawn the feeder
    /// thread. Returns `None` (with a log line) if no output device or
    /// compatible stream config can be opened — Mario mode still works
    /// silently in that case.
    pub fn start() -> Option<Self> {
        let ring = Arc::new(RingBuffer::new());
        let stop = Arc::new(AtomicBool::new(false));

        let stream = build_stream(ring.clone())?;
        // cpal streams start paused on some backends; kick it explicitly.
        if let Err(e) = stream.play() {
            tracing::warn!("cpal stream failed to start: {e}");
            return None;
        }

        let feeder = spawn_feeder(ring.clone(), stop.clone());

        Some(Self {
            ring,
            stream: Some(stream),
            feeder: Some(feeder),
            stop,
        })
    }

    /// Number of stereo frames currently queued in the ring buffer.
    /// Exposed so callers could throttle, but primarily documents the
    /// value passed to `sm64_audio_tick`.
    #[allow(dead_code)]
    fn queued_frames(&self) -> u32 {
        (self.ring.read_space() / SM64_CHANNELS as usize) as u32
    }

    /// Stop the audio stream and feeder thread. Idempotent: safe to call
    /// explicitly (e.g. before dropping the SM64 globals the feeder ticks
    /// against) and again from `Drop`.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }
        if let Some(handle) = self.feeder.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Sm64Audio {
    fn drop(&mut self) {
        self.stop();
        tracing::info!("SM64 audio stopped");
    }
}

// ── cpal stream construction ───────────────────────────────────────────

/// Build the cpal output stream, converting from our internal i16 ring to
/// whatever sample format the default device prefers. Prefers a 32 kHz /
/// 2-channel config; if the device won't do 32 kHz exactly we fall back to
/// its default config (the audio still plays, just at the device's native
/// rate — pitch shifts, but it does not stutter or crash).
fn build_stream(ring: Arc<RingBuffer>) -> Option<cpal::Stream> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;
    tracing::info!(
        "audio output device: {}",
        device.name().unwrap_or_else(|_| "<unknown>".into())
    );

    // Try to find a 32 kHz stereo config the device actually supports.
    let (config, sample_format) = match pick_config(&device) {
        Some(c) => c,
        None => {
            tracing::warn!(
                "no 32 kHz stereo output config; falling back to device default \
                 (audio may be pitch-shifted)"
            );
            let default = device.default_output_config().ok()?;
            let sample_format = default.sample_format();
            let config: StreamConfig = default.into();
            (config, sample_format)
        }
    };

    tracing::info!(
        "audio stream: {} Hz, {} ch, {:?}",
        config.sample_rate.0,
        config.channels,
        sample_format
    );

    let err_fn = |err| tracing::error!("cpal audio stream error: {err}");

    let stream = match sample_format {
        SampleFormat::I16 => device
            .build_output_stream(&config, move |data, _| write_i16(&ring, data), err_fn, None),
        SampleFormat::F32 => device
            .build_output_stream(&config, move |data, _| write_f32(&ring, data), err_fn, None),
        SampleFormat::U16 => device
            .build_output_stream(&config, move |data, _| write_u16(&ring, data), err_fn, None),
        other => {
            tracing::warn!("unsupported cpal sample format {other:?}; audio disabled");
            return None;
        }
    }
    .map_err(|e| {
        tracing::error!("cpal build_output_stream failed: {e}");
        e
    })
    .ok()?;

    Some(stream)
}

/// Pick a supported output config at 32 kHz with 2 channels, preferring
/// I16 then F32 then whatever is available. Returns the concrete
/// `StreamConfig` plus the sample format to build the stream with.
fn pick_config(
    device: &cpal::Device,
) -> Option<(StreamConfig, SampleFormat)> {
    let mut configs = device.supported_output_configs().ok()?;
    // Prefer 32 kHz / 2-channel, I16 first, then F32, then any format.
    let mut best: Option<(StreamConfig, SampleFormat)> = None;
    while let Some(range) = configs.next() {
        if range.channels() != SM64_CHANNELS {
            continue;
        }
        let rate = range.min_sample_rate().0;
        let max = range.max_sample_rate().0;
        if !(rate..=max).contains(&SM64_SAMPLE_RATE) {
            continue;
        }
        let cfg = range
            .with_sample_rate(cpal::SampleRate(SM64_SAMPLE_RATE));
        let fmt = cfg.sample_format();
        let sc: StreamConfig = cfg.into();
        // Prefer I16 (no conversion), then F32, then anything.
        match fmt {
            SampleFormat::I16 => return Some((sc, fmt)),
            SampleFormat::F32 if best.is_none() => best = Some((sc, fmt)),
            _ if best.is_none() => best = Some((sc, fmt)),
            _ => {}
        }
    }
    best
}

/// Drain the ring into an i16 output buffer, zero-filling the tail on
/// underrun so the device always gets a full frame of valid samples.
fn write_i16(ring: &RingBuffer, data: &mut [i16]) {
    let n = ring.pop(data);
    for s in data[n..].iter_mut() {
        *s = 0;
    }
}

/// Drain the ring, converting i16 → f32, silence-filling the tail.
fn write_f32(ring: &RingBuffer, data: &mut [f32]) {
    // Reuse a small stack buffer to batch pops and avoid per-sample work.
    let mut tmp = [0i16; 256];
    let mut filled = 0;
    while filled < data.len() {
        let want = (data.len() - filled).min(tmp.len());
        let got = ring.pop(&mut tmp[..want]);
        if got == 0 {
            break;
        }
        for &s in &tmp[..got] {
            data[filled] = f32::from_sample(s);
            filled += 1;
        }
        if got < want {
            break; // underrun
        }
    }
    for s in data[filled..].iter_mut() {
        *s = f32::EQUILIBRIUM;
    }
}

/// Drain the ring, converting i16 → u16, silence-filling the tail.
fn write_u16(ring: &RingBuffer, data: &mut [u16]) {
    let mut tmp = [0i16; 256];
    let mut filled = 0;
    while filled < data.len() {
        let want = (data.len() - filled).min(tmp.len());
        let got = ring.pop(&mut tmp[..want]);
        if got == 0 {
            break;
        }
        for &s in &tmp[..got] {
            data[filled] = u16::from_sample(s);
            filled += 1;
        }
        if got < want {
            break;
        }
    }
    for s in data[filled..].iter_mut() {
        *s = u16::EQUILIBRIUM;
    }
}

// ── Feeder thread ──────────────────────────────────────────────────────

/// Spawn the background thread that repeatedly calls `sm64_audio_tick` and
/// pushes the result into the ring, exiting when `stop` is set.
fn spawn_feeder(ring: Arc<RingBuffer>, stop: Arc<AtomicBool>) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name("sm64-audio-feeder".into())
        .spawn(move || feeder_loop(&ring, &stop))
        .expect("spawn sm64-audio-feeder")
}

fn feeder_loop(ring: &RingBuffer, stop: &AtomicBool) {
    let mut tick_buf = [0i16; TICK_BUFFER_LEN];
    while !stop.load(Ordering::Acquire) {
        // Frames (stereo pairs) currently queued, matching the SDL example's
        // `SDL_GetQueuedAudioSize(dev)/4` — the count is in *frames*, not
        // individual i16 values.
        let queued_frames = (ring.read_space() / SM64_CHANNELS as usize) as u32;

        // SAFETY: `sm64_audio_tick` writes at most `4 * num_audio_samples`
        // i16 values into `tick_buf` (≤ TICK_BUFFER_LEN). It reads no
        // pointer other than our buffer. Called from this dedicated thread,
        // exactly as the upstream SDL example does, concurrent with the
        // main thread's `sm64_mario_tick`.
        let per_channel = unsafe {
            ffi::sm64_audio_tick(queued_frames, NUM_DESIRED_SAMPLES, tick_buf.as_mut_ptr())
        };

        if per_channel > 0 {
            // Total i16 values produced: 2 channels × 2 × per_channel
            // (L and R each get `2 * per_channel` slots in the interleaved
            // layout written by `create_next_audio_buffer`).
            let total = (per_channel as usize) * 2 * 2;
            let total = total.min(TICK_BUFFER_LEN);
            ring.push(&tick_buf[..total]);
        }

        // Cooperative cadence; `stop` is checked at the top each iteration.
        // Sleep in small slices so drop responds promptly.
        let mut remaining = FEEDER_SLEEP;
        while !stop.load(Ordering::Acquire) && !remaining.is_zero() {
            let step = remaining.min(Duration::from_millis(5));
            thread::sleep(step);
            remaining = remaining.saturating_sub(step);
        }
    }
}
