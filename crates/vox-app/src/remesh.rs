//! Threaded chunk remeshing with stale-result protection.
//!
//! Edits mark chunks dirty; each frame the queue extracts slabs on the main
//! thread (cheap copies of live data), meshes them on rayon workers, and
//! uploads finished meshes. Every job carries the generation of the chunk at
//! dispatch time — results older than the chunk's latest generation are
//! dropped and the chunk re-queued, so an upload can never clobber a newer
//! edit.

use std::sync::mpsc::{Receiver, Sender, channel};

use glam::{IVec3, Vec3};
use vox_core::consts::CHUNK_SIZE;
use vox_core::{chunk_origin, voxel_center_m};
use vox_mesh::{MeshData, VoxelSlab, mesh_slab};
use vox_render::{Gpu, VoxelPipeline};
use vox_world::World;

/// Maximum mesh jobs dispatched per frame.
const MAX_DISPATCH_PER_FRAME: usize = 64;

type MeshResult = (IVec3, u64, MeshData);

/// Book-keeping for background remeshing.
pub struct RemeshQueue {
    /// Chunks awaiting dispatch, with the generation that queued them.
    pending: vox_core::FxHashMap<IVec3, u64>,
    /// Latest known generation per chunk; only matching results upload.
    latest: vox_core::FxHashMap<IVec3, u64>,
    counter: u64,
    tx: Sender<MeshResult>,
    rx: Receiver<MeshResult>,
    /// Jobs currently on workers (HUD stat).
    pub in_flight: usize,
}

impl Default for RemeshQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RemeshQueue {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            pending: Default::default(),
            latest: Default::default(),
            counter: 0,
            tx,
            rx,
            in_flight: 0,
        }
    }

    /// Number of chunks waiting to be dispatched.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Absorb the world's dirty set, bumping generations.
    pub fn absorb_dirty(&mut self, world: &mut World) {
        for key in world.drain_dirty() {
            self.counter += 1;
            self.pending.insert(key, self.counter);
            self.latest.insert(key, self.counter);
        }
    }

    /// Dispatch up to the per-frame budget, nearest chunks first.
    pub fn dispatch(&mut self, world: &World, camera_pos: Vec3) {
        if self.pending.is_empty() {
            return;
        }
        let s = world.cfg.voxel_size_m;
        let mut keys: Vec<IVec3> = self.pending.keys().copied().collect();
        keys.sort_by(|a, b| {
            let da = (voxel_center_m(chunk_origin(*a), s) - camera_pos).length_squared();
            let db = (voxel_center_m(chunk_origin(*b), s) - camera_pos).length_squared();
            da.total_cmp(&db)
        });
        for key in keys.into_iter().take(MAX_DISPATCH_PER_FRAME) {
            let generation = self
                .pending
                .remove(&key)
                .expect("key came from pending set");
            let origin = chunk_origin(key);
            let slab = VoxelSlab::extract(world, origin, IVec3::splat(CHUNK_SIZE as i32));
            let tx = self.tx.clone();
            self.in_flight += 1;
            rayon::spawn(move || {
                let mesh = mesh_slab(&slab, origin);
                // Receiver dropped only on shutdown; ignore send failure.
                let _ = tx.send((key, generation, mesh));
            });
        }
    }

    /// Upload finished meshes; requeue any that raced with a newer edit.
    pub fn collect(&mut self, gpu: &Gpu, pipeline: &mut VoxelPipeline) -> usize {
        let mut uploaded = 0;
        while let Ok((key, generation, mesh)) = self.rx.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            let latest = self.latest.get(&key).copied().unwrap_or(0);
            if generation == latest {
                pipeline.upload_chunk(gpu, key, &mesh);
                uploaded += 1;
            }
            // Older than latest: a newer job is pending or in flight; drop.
        }
        uploaded
    }
}
