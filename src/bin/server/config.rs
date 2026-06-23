use serde::Deserialize;
use tractile::config::Configuration;

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    port: Option<u16>,
    pub embedding: Option<Configuration>,
}

impl ServerConfig {
    pub fn read(path: &str) -> tractile::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    /// Port resolution order: `server.toml` → `PORT` env var → 80.
    pub fn port(&self) -> u16 {
        self.port
            .or_else(|| std::env::var("PORT").ok().and_then(|s| s.parse().ok()))
            .unwrap_or(80)
    }
}
