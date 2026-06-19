use serde::Deserialize;
use tractile::config::Configuration;

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "ServerConfig::default_port")]
    port: u16,
    pub embedding: Option<Configuration>,
}

impl ServerConfig {
    pub fn read(path: &str) -> tractile::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    fn default_port() -> u16 {
        8080
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}
