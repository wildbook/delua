#[derive(shrinkwraprs::Shrinkwrap, Clone)]
#[shrinkwrap(mut)]
pub struct IgnoreDebug<T: Clone>(pub T);

impl<T: Clone> std::fmt::Debug for IgnoreDebug<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "..")
    }
}

pub fn discard<T>(_: T) {}
