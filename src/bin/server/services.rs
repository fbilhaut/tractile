use std::time::Instant;

use actix_web::*;
use serde::{Deserialize, Serialize};

use super::context::AppContext;

#[derive(Deserialize)]
pub struct EmbedInput {
    texts: Vec<String>,
}

#[derive(Serialize)]
pub struct EmbedOutput {
    vectors: Vec<Vec<f32>>,
}

/// Health check required by RunPod load balancing.
/// Returns 204 while the model is still loading, 200 once ready.
#[get("/ping")]
pub async fn ping(ctx: web::Data<AppContext>) -> impl Responder {
    if ctx.is_ready() {
        HttpResponse::Ok().json(serde_json::json!({"status": "healthy"}))
    } else {
        HttpResponse::NoContent().finish()
    }
}

#[post("/embed")]
pub async fn extract_embeddings(
    context: web::Data<AppContext>,
    input: web::Json<EmbedInput>,
) -> Result<impl Responder> {
    let guard = context.embedding_pipeline.read().unwrap();
    let Some(pipeline) = guard.as_ref() else {
        return Err(error::ErrorServiceUnavailable("model is still loading"));
    };
    let n = input.texts.len();
    log::debug!("Embedding {n} text(s)");
    let texts: Vec<&str> = input.texts.iter().map(String::as_str).collect();
    let t = Instant::now();
    let output = pipeline
        .embed_texts(&texts)
        .map_err(error::ErrorInternalServerError)?;
    log::debug!("Done in {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    let vectors = (0..output.len()).map(|i| output.embedding(i).to_vec()).collect();
    Ok(web::Json(EmbedOutput { vectors }))
}
