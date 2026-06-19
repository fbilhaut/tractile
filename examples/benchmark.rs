/// Measures throughput (sentences/sec) for a fixed batch size over many runs.
/// Run with `--release` for meaningful numbers.
use std::time::Instant;
use tractile::{Configuration, ExtractorMode, TextEmbeddingPipeline};

const TOKENIZER: &str =
    "/Users/fbilhaut/src/perso/open-source/gte-rs/models/gte-multilingual-base/tokenizer.json";
const MODEL: &str =
    "/Users/fbilhaut/src/perso/open-source/gte-rs/models/gte-multilingual-base/onnx/model.onnx";

const BATCH_SIZE: usize = 32;
const N_WARMUP: usize = 3;
const N_RUNS: usize = 50;

const SEED_TEXTS: &[&str] = &[
    "What is the capital of France?",
    "How to implement quick sort in python?",
    "Die Hauptstadt von Frankreich ist Paris.",
    "La capital de Francia es París.",
    "London is the capital of the UK",
    "Rust is a systems programming language focused on safety.",
    "The transformer architecture revolutionized natural language processing.",
    "Neural networks learn representations from raw data.",
    "Le chat est sur le tapis.",
    "机器学习是人工智能的一个分支。",
];

fn main() -> tractile::Result<()> {
    let config = Configuration::new(TOKENIZER, MODEL)
        .with_output_index(1)
        .with_mode(ExtractorMode::Raw);

    let pipeline = TextEmbeddingPipeline::new(config)?;

    let batch: Vec<&str> = (0..BATCH_SIZE).map(|i| SEED_TEXTS[i % SEED_TEXTS.len()]).collect();

    println!("Batch size : {BATCH_SIZE}");
    println!("Warmup runs: {N_WARMUP}");
    println!("Bench runs : {N_RUNS}");

    for _ in 0..N_WARMUP {
        pipeline.embed_texts(&batch)?;
    }

    let t0 = Instant::now();
    for _ in 0..N_RUNS {
        pipeline.embed_texts(&batch)?;
    }
    let elapsed = t0.elapsed();

    let total = N_RUNS * BATCH_SIZE;
    let avg_ms = elapsed.as_millis() as f64 / N_RUNS as f64;
    let throughput = total as f64 / elapsed.as_secs_f64();

    println!("Total time : {elapsed:.2?}");
    println!("Avg / batch: {avg_ms:.1} ms");
    println!("Throughput : {throughput:.0} sentences/sec");

    Ok(())
}
