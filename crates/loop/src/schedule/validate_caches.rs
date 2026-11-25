//! Checks the inputs staged in shared memory against the final nest and
//! the target: where each stage sits, that each pipelined loop has tiles to
//! fill ahead, that each reindexed input is staged, that each swizzle
//! permutes bytes only within its tile, and that the tiles fit.

use super::apply::cache::{LoadSite, loads_of};
use super::error::ScheduleError;
use super::loop_tree::{CacheRead, LoopAnnotation, LoopKind, LoopTree};
use crate::compute::ComputeDef;
use crate::lower::{slots_at, tile_extents};
use ptxc_core::{ByteSize, Dim};
use ptxc_target::Target;

/// Each cache's tile extents, per axis of its input, once every stage sits
/// where lowering can fill it and the tiles fit the block's shared memory.
pub(super) fn check_caches(
    compute: &ComputeDef,
    tree: &LoopTree,
    target: &Target,
) -> Result<Vec<Vec<Dim>>, ScheduleError> {
    for cache in tree.caches() {
        check_placement(compute, tree, cache)?;
    }
    check_pipelines(tree)?;
    check_reindexed(compute, tree)?;
    let tiles = tile_extents(compute, tree)?;
    check_swizzles(compute, tree, &tiles)?;
    check_shared_memory(compute, tree, &tiles, target)?;
    Ok(tiles)
}

/// `cache` is at a serial loop of the nest, in a kernel whose reduction
/// stays within each thread. At or inside the reduction's loops, which run
/// only around the reduction's body, its input is read nowhere else.
fn check_placement(
    compute: &ComputeDef,
    tree: &LoopTree,
    cache: &CacheRead,
) -> Result<(), ScheduleError> {
    let name = || compute.inputs()[cache.tensor].name.clone();
    let at = &tree.loops[cache.at];
    if !tree.nest.contains(&cache.at) || !at.annotation.runs_serially() {
        return Err(ScheduleError::CacheNotSerial { tensor: name(), at: at.name.clone() });
    }
    let shares_reduction = tree.nest.iter().any(|&id| {
        let found = &tree.loops[id];
        found.kind == LoopKind::Reduce && matches!(found.annotation, LoopAnnotation::Bound(_))
    });
    if shares_reduction {
        return Err(ScheduleError::CacheCrossThread(name()));
    }
    let mut enclosing = tree.nest.iter().take_while(|&&id| id != cache.at);
    let in_reduction_loops =
        at.kind == LoopKind::Reduce || enclosing.any(|&id| tree.loops[id].kind == LoopKind::Reduce);
    let loads = loads_of(compute, cache.tensor);
    if in_reduction_loops && loads.iter().any(|load| load.site == LoadSite::OutsideReduction) {
        return Err(ScheduleError::CacheOutsideReduction(name()));
    }
    Ok(())
}

/// Every pipelined loop has an input staged at it.
fn check_pipelines(tree: &LoopTree) -> Result<(), ScheduleError> {
    for &id in &tree.nest {
        let found = &tree.loops[id];
        let pipelined = matches!(found.annotation, LoopAnnotation::Pipelined(_));
        if pipelined && !tree.caches().iter().any(|cache| cache.at == id) {
            return Err(ScheduleError::PipelineUnstaged(found.name.clone()));
        }
    }
    Ok(())
}

/// Every reindexed input is staged.
fn check_reindexed(compute: &ComputeDef, tree: &LoopTree) -> Result<(), ScheduleError> {
    for found in tree.reindexed() {
        if !tree.caches().iter().any(|cache| cache.tensor == found.tensor) {
            return Err(ScheduleError::ReindexUnstaged(
                compute.inputs()[found.tensor].name.clone(),
            ));
        }
    }
    Ok(())
}

/// Each swizzled tile holds whole blocks of its swizzle, so the swizzle
/// moves bytes only within the tile. A unit, at least 16 bytes, always
/// holds whole elements.
fn check_swizzles(
    compute: &ComputeDef,
    tree: &LoopTree,
    tiles: &[Vec<Dim>],
) -> Result<(), ScheduleError> {
    for (cache, extents) in tree.caches().iter().zip(tiles) {
        let Some(swizzle) = cache.swizzle else { continue };
        let bytes = tile_bytes(compute, cache, extents);
        if !bytes.is_some_and(|bytes| bytes.is_multiple_of(swizzle.block_bytes())) {
            return Err(ScheduleError::SwizzleLayout(compute.inputs()[cache.tensor].name.clone()));
        }
    }
    Ok(())
}

/// The tiles together, each as many times as its loop's pipeline has
/// stages, fit the shared memory a block has.
fn check_shared_memory(
    compute: &ComputeDef,
    tree: &LoopTree,
    tiles: &[Vec<Dim>],
    target: &Target,
) -> Result<(), ScheduleError> {
    let mut bytes: u64 = 0;
    for (cache, extents) in tree.caches().iter().zip(tiles) {
        let slots = slots_at(tree, cache.at).get();
        let tile = tile_bytes(compute, cache, extents).and_then(|tile| tile.checked_mul(slots));
        bytes =
            tile.and_then(|tile| bytes.checked_add(tile)).ok_or(ScheduleError::ExtentOverflow)?;
    }
    let limit = target.limits().block.smem;
    if bytes > u64::from(limit.get()) {
        return Err(ScheduleError::SharedTooLarge { bytes: ByteSize::new(bytes), limit });
    }
    Ok(())
}

/// The bytes of one copy of `cache`'s tile of `extents`, if they fit `u64`.
fn tile_bytes(compute: &ComputeDef, cache: &CacheRead, extents: &[Dim]) -> Option<u64> {
    let element = compute.inputs()[cache.tensor].dtype.element_size().get();
    extents.iter().try_fold(element, |size, dim| size.checked_mul(dim.get()))
}
