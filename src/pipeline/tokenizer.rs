use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::config::Configuration;

pub(super) fn load_tokenizer(config: &Configuration) -> crate::Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(config.tokenizer_path())
        .map_err(|e| crate::Error::from(e.to_string()))?;

    if let Some(max_length) = config.max_length() {
        tokenizer
            .with_truncation(Some(TruncationParams { max_length, ..Default::default() }))
            .map_err(|e| crate::Error::from(e.to_string()))?;
    }

    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    Ok(tokenizer)
}
