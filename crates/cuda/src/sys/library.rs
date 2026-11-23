//! Our own `dlopen`, `dlsym` and `dlerror` declarations (D24). Libraries
//! are opened once and never closed.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;

/// `RTLD_NOW` from glibc's `dlfcn.h`: resolve every symbol at load.
const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

/// An opened shared library, kept loaded for the rest of the process.
/// Besides libcuda, `ptxc-bench` opens its baseline library with it (D24,
/// D55).
#[derive(Debug)]
pub struct Library(NonNull<c_void>);

impl Library {
    /// Opens `name`, or returns `dlerror`'s message.
    pub fn open(name: &CStr) -> Result<Self, String> {
        // SAFETY: `name` is NUL-terminated; dlopen has no other precondition.
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        NonNull::new(handle).map(Self).ok_or_else(last_error)
    }

    /// The address of `symbol`, or `dlerror`'s message.
    pub fn symbol(&self, symbol: &CStr) -> Result<NonNull<c_void>, String> {
        // SAFETY: the handle came from dlopen and is never closed, and
        // `symbol` is NUL-terminated.
        let address = unsafe { dlsym(self.0.as_ptr(), symbol.as_ptr()) };
        NonNull::new(address)
            .ok_or_else(|| format!("{}: {}", symbol.to_string_lossy(), last_error()))
    }
}

/// `dlerror`'s description of the last failure on this thread.
fn last_error() -> String {
    // SAFETY: dlerror has no preconditions.
    let message = unsafe { dlerror() };
    if message.is_null() {
        return "unknown dynamic loader error".to_owned();
    }
    // SAFETY: a non-null dlerror result is a NUL-terminated string that stays
    // valid until the next dl* call on this thread, and it is copied here.
    unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
}
