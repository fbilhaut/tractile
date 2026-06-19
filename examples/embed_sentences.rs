/// Embeds a small batch of multilingual sentences and prints pairwise cosine
/// similarities against the first one.  Mirrors `embed-mul-sentence.rs` from gte-rs.
use tractile::{Configuration, ExtractorMode, TextEmbeddingPipeline, TextInput};

const TOKENIZER: &str =
    "/Users/fbilhaut/src/perso/open-source/gte-rs/models/gte-multilingual-base/tokenizer.json";
const MODEL: &str =
    "/Users/fbilhaut/src/perso/open-source/gte-rs/models/gte-multilingual-base/onnx/model.onnx";

fn main() -> tractile::Result<()> {
    let config = Configuration::new(TOKENIZER, MODEL)
        .with_output_index(1)
        .with_mode(ExtractorMode::Raw);

    let pipeline = TextEmbeddingPipeline::new(config)?;

    let input = TextInput::from_str(&[
        "What is the capital of France?",
        "How to implement quick sort in python?",
        "Die Hauptstadt von Frankreich ist Paris.",
        "La capital de Francia es París.",
        "London is the capital of the UK",
    ]);

    let output = pipeline.embed(&input)?;
    let distances = output.distances_to_first();

    println!("Distances to {:?}:", input.texts()[0]);
    for (text, dist) in input.texts()[1..].iter().zip(distances.iter().skip(1)) {
        println!("  {dist:.4}  {text:?}");
    }

    Ok(())
}
