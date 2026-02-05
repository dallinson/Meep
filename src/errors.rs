use core::fmt;
use std::error::Error;

#[derive(Debug)]
pub(crate) struct SliceSizesNotRatiosError {
    pub(crate) description: String,
}

impl Error for SliceSizesNotRatiosError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn Error> {
        self.source()
    }
}

impl fmt::Display for SliceSizesNotRatiosError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description)
    }
}
