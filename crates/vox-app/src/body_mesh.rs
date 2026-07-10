//! Threaded debris body meshing, mirroring `RemeshQueue`'s pattern for
//! chunks: mesh generation is dispatched to rayon workers and collected (and
//! uploaded to the GPU) on the main thread once a frame.
//!
//! Simpler than chunk remeshing: a body's geometry is fixed forever once
//! it's spawned (destroying it further despawns it and spawns brand new
//! bodies with their own mesh jobs), so there's no generation/staleness
//! tracking to worry about -- just dispatch once and collect whenever the
//! result arrives, however many frames that takes.

use std::sync::mpsc::{Receiver, Sender, channel};

use glam::IVec3;
use vox_mesh::{MeshData, VoxelSlab, mesh_slab};
use vox_render::{BodyMeshKey, Gpu, VoxelPipeline};
use vox_world::Voxel;

type BodyMeshResult = (BodyMeshKey, MeshData);

/// Book-keeping for background debris body meshing.
pub struct BodyMeshQueue {
    tx: Sender<BodyMeshResult>,
    rx: Receiver<BodyMeshResult>,
    /// Jobs currently on workers (HUD stat, mirrors `RemeshQueue::in_flight`).
    pub in_flight: usize,
}

impl Default for BodyMeshQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl BodyMeshQueue {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            tx,
            rx,
            in_flight: 0,
        }
    }

    /// Dispatch a newly-spawned body's mesh generation to a worker thread.
    /// `voxels` is a copy of the body's own grid (bodies are small -- a few
    /// voxels to a few hundred thousand at the very largest -- so cloning it
    /// once to hand across the thread boundary is far cheaper than the
    /// meshing work itself, and avoids needing any lock on `PhysicsWorld`).
    pub fn dispatch(&mut self, key: BodyMeshKey, dims: IVec3, voxels: Vec<Voxel>) {
        let tx = self.tx.clone();
        self.in_flight += 1;
        rayon::spawn(move || {
            let slab = VoxelSlab::from_grid(dims, &voxels);
            // Zero seed: a body has no meaningful "world origin" (it moves),
            // so the jitter pattern is anchored to its own local grid only.
            let mesh = mesh_slab(&slab, IVec3::ZERO);
            // Receiver dropped only on shutdown; ignore send failure.
            let _ = tx.send((key, mesh));
        });
    }

    /// Upload every mesh that's finished since the last call, returning the
    /// keys that just arrived so the caller can resolve anything waiting on
    /// them (see `VoxApp::replace_body`'s "keep the old ghost mesh until its
    /// replacement is ready" bookkeeping). A result for a body that was
    /// despawned (e.g. evicted by the debris budget, or destroyed again)
    /// before its mesh even arrived is uploaded anyway and then simply never
    /// referenced again by any live `BodyId` (`BodyMeshKey` bakes in the
    /// generation, so a reused slot never collides with a stale result) --
    /// a small, bounded, one-off leaked GPU buffer set in that narrow race,
    /// not a correctness problem.
    pub fn collect(&mut self, gpu: &Gpu, pipeline: &mut VoxelPipeline) -> Vec<BodyMeshKey> {
        let mut uploaded = Vec::new();
        while let Ok((key, mesh)) = self.rx.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            pipeline.upload_body(gpu, key, &mesh);
            uploaded.push(key);
        }
        uploaded
    }
}
