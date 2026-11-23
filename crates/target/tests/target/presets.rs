//! The built-in targets.

use ptxc_target::{
    BlockCount, PtxIsaVersion, RegisterCount, SmVersion, SmemBytes, ThreadCount, ThreadsPerBlock,
    UnsupportedSm, preset,
};

#[test]
fn sm_89_declares_the_first_isa_with_its_fp8_instructions() {
    let target = preset(SmVersion::SM_89).unwrap();

    assert_eq!(target.sm(), SmVersion::SM_89);
    assert_eq!(target.ptx_isa(), PtxIsaVersion::new(8, 4));
    assert_eq!(target.mma(), &[]);
}

#[test]
fn sm_89_has_the_limits_its_driver_reports() {
    let limits = *preset(SmVersion::SM_89).unwrap().limits();

    assert_eq!(limits.warp_size, ThreadCount::new(32));
    assert_eq!(limits.registers_per_thread, RegisterCount::new(255));
    assert_eq!(limits.block.threads, ThreadsPerBlock::new(1024));
    assert_eq!(limits.block.registers, RegisterCount::new(65_536));
    assert_eq!(limits.block.smem, SmemBytes::new(48 * 1024));
    assert_eq!(limits.block.smem_opt_in, SmemBytes::new(99 * 1024));
    assert_eq!(limits.block.smem_reserved, SmemBytes::new(1024));
    assert_eq!(limits.sm.threads, ThreadCount::new(1536));
    assert_eq!(limits.sm.blocks, BlockCount::new(24));
    assert_eq!(limits.sm.registers, RegisterCount::new(65_536));
    assert_eq!(limits.sm.smem, SmemBytes::new(100 * 1024));
    let block_extents = limits.block.extents.map(ThreadsPerBlock::get);
    assert_eq!(block_extents, [1024, 1024, 64]);
    assert_eq!(limits.grid.extents.map(BlockCount::get), [2_147_483_647, 65_535, 65_535]);
}

#[test]
fn sm_89_limits_are_consistent_with_each_other() {
    let limits = *preset(SmVersion::SM_89).unwrap().limits();
    let (block, sm) = (limits.block, limits.sm);

    assert_eq!(block.threads.get() % limits.warp_size.get(), 0);
    assert!(block.threads.get() <= sm.threads.get());
    assert!(block.smem <= block.smem_opt_in);
    assert_eq!(block.smem_opt_in.get() + block.smem_reserved.get(), sm.smem.get());
    assert!(block.registers <= sm.registers);
    assert!(limits.registers_per_thread.get() * limits.warp_size.get() <= block.registers.get());
}

#[test]
fn an_sm_without_a_preset_is_unsupported_and_names_the_supported_ones() {
    let error = preset(SmVersion::new(8, 6)).unwrap_err();

    assert_eq!(error, UnsupportedSm(SmVersion::new(8, 6)));
    assert_eq!(error.to_string(), "ptxc has no target for sm_86; supported: sm_89");
}
