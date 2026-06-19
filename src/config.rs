use std::path::{Path, PathBuf};
use crate::pipeline::ExtractorMode;

/// All model-specific parameters needed to build a [`TextEmbeddingPipeline`].
///
/// [`TextEmbeddingPipeline`]: crate::TextEmbeddingPipeline
#[derive(Debug, Clone)]
pub struct Configuration {
    tokenizer_path: PathBuf,
    model_path: PathBuf,
    /// Which ONNX output index contains the embeddings.
    output_index: usize,
    mode: ExtractorMode,
    max_length: Option<usize>,
}

impl Configuration {
    pub fn new(tokenizer_path: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            tokenizer_path: tokenizer_path.into(),
            model_path: model_path.into(),
            output_index: 0,
            mode: ExtractorMode::default(),
            max_length: None,
        }
    }

    pub fn with_output_index(mut self, index: usize) -> Self {
        self.output_index = index;
        self
    }

    pub fn with_mode(mut self, mode: ExtractorMode) -> Self {
        self.mode = mode;
        self
    }

    /// Truncate input sequences to at most `max_length` tokens.
    pub fn with_max_length(mut self, max_length: Option<usize>) -> Self {
        self.max_length = max_length;
        self
    }

    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn output_index(&self) -> usize {
        self.output_index
    }

    pub fn mode(&self) -> &ExtractorMode {
        &self.mode
    }

    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }
}
