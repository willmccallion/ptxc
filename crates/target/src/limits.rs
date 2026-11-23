//! Hardware limits as newtypes: threads, blocks, registers and shared memory,
//! per thread, per block and per SM.

/// A number of threads in a block.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ThreadsPerBlock(u32);

/// A number of threads that is not a block size: a warp, or the threads
/// resident on an SM.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ThreadCount(u32);

/// A number of thread blocks.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BlockCount(u32);

/// A number of 32-bit registers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RegisterCount(u32);

/// An amount of shared memory in bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SmemBytes(u32);

macro_rules! count_methods {
    ($($name:ident),*) => {$(
        impl $name {
            /// Wraps a count.
            pub const fn new(count: u32) -> Self {
                Self(count)
            }

            /// The count.
            pub const fn get(self) -> u32 {
                self.0
            }
        }
    )*};
}

count_methods!(ThreadsPerBlock, ThreadCount, BlockCount, RegisterCount, SmemBytes);

/// What one thread block may use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockLimits {
    /// The most threads in a block.
    pub threads: ThreadsPerBlock,
    /// The most threads along each of the block's x, y and z axes.
    pub extents: [ThreadsPerBlock; 3],
    /// The most registers a block's threads hold together.
    pub registers: RegisterCount,
    /// The most shared memory without opting in, static and dynamic.
    pub smem: SmemBytes,
    /// The most dynamic shared memory after opting in with
    /// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`.
    pub smem_opt_in: SmemBytes,
    /// The shared memory the driver reserves for each resident block, on top
    /// of what the block asks for.
    pub smem_reserved: SmemBytes,
}

/// What the blocks resident on one SM may use together, which bounds
/// occupancy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SmLimits {
    /// The most resident threads.
    pub threads: ThreadCount,
    /// The most resident blocks.
    pub blocks: BlockCount,
    /// The registers in the SM's register file.
    pub registers: RegisterCount,
    /// The SM's shared memory, including each block's reserved part.
    pub smem: SmemBytes,
}

/// How large a grid may be.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GridLimits {
    /// The most blocks along each of the grid's x, y and z axes.
    pub extents: [BlockCount; 3],
}

/// A target's hardware limits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Limits {
    /// The threads in a warp.
    pub warp_size: ThreadCount,
    /// The most registers one thread may use.
    pub registers_per_thread: RegisterCount,
    /// Per-block limits.
    pub block: BlockLimits,
    /// Per-SM limits.
    pub sm: SmLimits,
    /// Grid limits.
    pub grid: GridLimits,
}
