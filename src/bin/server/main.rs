mod config;
mod context;
mod services;

use actix_web::*;
use actix_web::middleware::Logger;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init();

    let config_path = std::env::args().nth(1).unwrap_or_else(|| "server.toml".to_string());
    let config = config::ServerConfig::read(&config_path).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    log::info!("Using config: {:?}", config);

    let port = config.port();
    let context = web::Data::new(context::AppContext::new(&config).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?,);

    log::info!("Listening on 0.0.0.0:{port}");

    HttpServer::new(move || {
        App::new()
            .app_data(context.clone())
            .service(services::extract_embeddings)
            .wrap(Logger::default())
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
