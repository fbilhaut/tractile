pub mod config;
pub mod input;
pub mod output;
pub mod pipeline;

pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T> = std::result::Result<T, Error>;
