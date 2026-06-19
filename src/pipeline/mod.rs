mod embeddings;
mod inference;
mod model;
mod tokenizer;

pub use embeddings::ExtractorMode;

use model::Model;
use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

use crate::config::Configuration;
use crate::input::TextInput;
use crate::output::TextEmbeddings;

/// A ready-to-use text embedding pipeline.
///
/// Wraps a tokenizer and an ONNX model loaded according to a [`Configuration`].
pub struct TextEmbeddingPipeline {
    model: Model,
    tokenizer: Tokenizer,
    config: Configuration,
}

impl TextEmbeddingPipeline {
    /// Build a pipeline from the given configuration.
    pub fn new(config: Configuration) -> crate::Result<Self> {
        let tokenizer = tokenizer::load_tokenizer(&config)?;
        let model = model::load_model(&config)?;
        Ok(Self { model, tokenizer, config })
    }

    /// Embed a [`TextInput`] batch.
    pub fn embed(&self, input: &TextInput) -> crate::Result<TextEmbeddings> {
        let texts: Vec<&str> = input.texts.iter().map(String::as_str).collect();
        self.embed_texts(&texts)
    }

    /// Convenience method — embed a slice of string references directly.
    pub fn embed_texts(&self, texts: &[&str]) -> crate::Result<TextEmbeddings> {
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| crate::Error::from(e.to_string()))?;

        let n = encodings.len();
        let seq_len = encodings[0].len();

        let flat_ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|&x| x as i64))
            .collect();
        let flat_mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|&x| x as i64))
            .collect();

        let ids: Tensor = tract_ndarray::Array2::from_shape_vec((n, seq_len), flat_ids)?.into();
        let mask: Tensor =
            tract_ndarray::Array2::from_shape_vec((n, seq_len), flat_mask.clone())?.into();

        let outputs = inference::run_inference(&self.model, ids, mask)?;

        let emb = embeddings::extract_embeddings(
            &outputs,
            self.config.output_index(),
            self.config.mode(),
            &flat_mask,
            n,
            seq_len,
        )?;
        Ok(TextEmbeddings { embeddings: emb })
    }
}
