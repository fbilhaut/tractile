/// A batch of texts to embed.
#[derive(Debug, Clone)]
pub struct TextInput {
    pub(crate) texts: Vec<String>,
}

impl TextInput {
    pub fn new(texts: Vec<String>) -> Self {
        Self { texts }
    }

    pub fn from_str(texts: &[&str]) -> Self {
        Self { texts: texts.iter().map(|s| s.to_string()).collect() }
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    pub fn texts(&self) -> &[String] {
        &self.texts
    }
}
