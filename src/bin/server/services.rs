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

#[post("/embed")]
pub async fn extract_embeddings(context: web::Data<AppContext>, input: web::Json<EmbedInput>) -> Result<impl Responder> {
    let Some(pipeline) = &context.embedding else {
        return Err(error::ErrorServiceUnavailable("no embedding model loaded"));
    };
    let texts: Vec<&str> = input.texts.iter().map(String::as_str).collect();
    let output = pipeline
        .lock()
        .unwrap()
        .embed_texts(&texts)
        .map_err(error::ErrorInternalServerError)?;

    let vectors = (0..output.len()).map(|i| output.embedding(i).to_vec()).collect();
    Ok(web::Json(EmbedOutput { vectors }))
}
