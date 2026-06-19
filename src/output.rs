use ndarray::{Array1, Array2, ArrayView1, Zip};

/// A batch of embeddings produced by [`TextEmbeddingPipeline`].
///
/// [`TextEmbeddingPipeline`]: crate::TextEmbeddingPipeline
pub struct TextEmbeddings {
    /// Shape: `[num_texts, embedding_dim]`.
    pub embeddings: Array2<f32>,
}

impl TextEmbeddings {
    /// View the embedding for text at `index`.
    pub fn embedding(&self, index: usize) -> ArrayView1<'_, f32> {
        self.embeddings.row(index)
    }

    /// Number of embeddings (equals the number of input texts).
    pub fn len(&self) -> usize {
        self.embeddings.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Cosine similarity between two embedding vectors.
    pub fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
        let dot = Zip::from(a).and(b).fold(0f32, |acc, &x, &y| acc + x * y);
        let norm_a = a.mapv(|x| x * x).sum().sqrt();
        let norm_b = b.mapv(|x| x * x).sum().sqrt();
        dot / (norm_a * norm_b)
    }

    /// Cosine similarity of every embedding against the one at `index`.
    pub fn distances_to(&self, index: usize) -> Array1<f32> {
        let reference = self.embeddings.row(index);
        self.embeddings.rows().into_iter().map(|row| Self::cosine_similarity(&reference, &row)).collect()
    }

    /// Cosine similarity of every embedding against the first one.
    pub fn distances_to_first(&self) -> Array1<f32> {
        self.distances_to(0)
    }
}
