# 🧲 tractile

Text-embedding inference in pure Rust powered by [tract](https://github.com/sonos/tract).

> NOTE: work in progress, alpha version, public API subject to change.

## Features

- Pure-Rust ONNX inference via `tract-onnx`
- HuggingFace-compatible tokenizer via `tokenizers`
- Optional Apple Metal acceleration (`--features metal`)
- Configurable embedding extraction: `Raw`, `Token(n)`, `MeanPool`
- `Configuration` is serializable/deserializable via `serde`

## Quick start

```rust
use tractile::config::{Configuration, ExtractorMode};
use tractile::input::TextInput;
use tractile::pipeline::TextEmbeddingPipeline;

let config = Configuration::new("path/to/tokenizer.json", "path/to/model.onnx")
    .with_output_index(1)           // index of the output to use
    .with_mode(ExtractorMode::Raw)  // sentence_embedding output is already [n, dim]
    .with_max_length(Some(512));    // optional truncation

let pipeline = TextEmbeddingPipeline::new(config)?;

let input = TextInput::from_str(&[
    "What is the capital of France?",
    "Paris est la capitale de la France.",
]);

let output = pipeline.embed(&input)?;
let similarities = output.distances_to_first(); // cosine similarity against first text
println!("{similarities}");
```

## Installation

```toml
[dependencies]
tractile = { path = "." }

# Enable Apple Metal (Apple Silicon only)
# tractile = { path = ".", features = ["metal"] }
```

## Configuration

`Configuration` can be built programmatically or deserialized from any serde-compatible format (JSON, TOML, …).

| Method | Default | Description |
|---|---|---|
| `with_output_index(usize)` | `0` | Which ONNX output to use |
| `with_mode(ExtractorMode)` | `Raw` | How to extract a per-sequence vector |
| `with_max_length(Option<usize>)` | `None` | Truncate sequences to this length |

### ExtractorMode

| Variant | Output shape | Use case |
|---|---|---|
| `Raw` | `[n, dim]` | Dedicated sentence-embedding output (e.g. gte-multilingual-base output 1) |
| `Token(i)` | `[n, seq, dim]` → take token `i` | CLS-token models: `Token(0)` |
| `MeanPool` | `[n, seq, dim]` → masked mean | Mean-pooling over attended tokens |

### Custom extractor

Implement the `tractile::pipeline::Extractor` trait to plug in your own extraction logic.

## Examples

### Sentence similarity

```sh
cargo run --release --example embed_sentences
```

Mirrors the `embed-mul-sentence.rs` example from [gte-rs](https://github.com/fbilhaut/gte-rs). Embeds a multilingual batch and prints cosine similarities against the first sentence.

### Benchmark

```sh
# CPU
cargo run --release --example benchmark

# Metal (Apple Silicon)
cargo run --release --example benchmark --features metal
```

Runs `N_WARMUP` warm-up passes then `N_RUNS` timed passes over a batch of `BATCH_SIZE` sentences and reports average latency and throughput.

## Platform support

| Platform | Status |
|---|---|
| macOS (Apple Silicon) | CPU + optional Metal |
| macOS (Intel) / Linux / Windows | CPU |
| CUDA | Planned |

## License

Apache-2.0
