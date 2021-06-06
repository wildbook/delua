/// Convenience wrapper that ignores a value's contents when printing using a derived Debug.
#[derive(shrinkwraprs::Shrinkwrap, Clone)]
#[shrinkwrap(mut)]
pub struct IgnoreDebug<T: Clone>(pub T);

impl<T: Clone> std::fmt::Debug for IgnoreDebug<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "..") }
}

/// Replacement for `drop` that explicitly discards a value, even if the value is either `Copy` or
/// a reference. This gets rid of clippy's complaints when we're doing this intentionally while not
/// removing the lint completely for if we'd happen to do it accidentally.
pub fn discard<T>(_: T) {}
