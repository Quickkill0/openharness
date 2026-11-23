pub mod activity;
pub mod transcript;
pub mod persistence;
pub mod manager;
pub mod settings;
pub mod sse;
mod reasoning;

pub use manager::{ChatHandle, ChatEvent, start};
