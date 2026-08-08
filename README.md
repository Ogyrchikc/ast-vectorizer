# AST Vectorizer 🚀

A high-performance Rust library for generating vector embeddings from text (like Abstract Syntax Trees, source code, or documentation) using HuggingFace models and ONNX Runtime.

Designed as **Module 3** in a larger AI-driven code analysis ecosystem, this library cleanly abstracts away the complexity of hardware acceleration and tokenization.

## Features ✨

* **Hardware Agnostic**: Automatically falls back gracefully: `NPU -> GPU -> CPU`.
* **Plug & Play**: Designed as a clean library (`lib.rs`) ready to be integrated into any Rust project.
* **Smart Pooling**: Supports multiple pooling strategies (`Mean`, `Cls`, `None`).
* **L2 Normalization**: Outputs are properly L2-normalized and ready for Cosine Similarity searches in vector databases.
* **Error Handling**: Uses `thiserror` for robust, easily debuggable errors.

## Usage 🛠️

First, make sure you have your HuggingFace ONNX model and tokenizer. By default, the examples expect the `all-MiniLM-L6-v2` model in a `models/` directory.

Add this library to your project, then you can use it like so:

```rust
use ast_vectorizer::{Vectorizer, VectorizerConfig, Device, PoolingStrategy};

fn main() {
    let config = VectorizerConfig {
        model_path: "models/all-MiniLM-L6-v2/model.onnx".to_string(),
        tokenizer_path: "models/all-MiniLM-L6-v2/tokenizer.json".to_string(),
        pooling: PoolingStrategy::Mean,
        device: Device::Auto, // Will try NPU, then GPU, then CPU
        use_token_type_ids: true,
    };

    let mut vectorizer = Vectorizer::new(config).expect("Failed to init");
    
    let sample_text = "def calculate_sum(a, b):\n    return a + b";
    let vector = vectorizer.embed(sample_text).expect("Failed to vectorize");

    println!("Success! Vector length: {}", vector.len());
}
```

## Running the Demo 🏃‍♂️

You can run the built-in demo to test your hardware and model configuration:

```bash
cargo run --example demo
```

*Note: If ONNX Runtime cannot find system drivers for your GPU/NPU, it will safely print an internal warning and fallback to CPU execution without crashing your app.*

## Testing 🧪

The module is fully covered by unit tests (initialization, vector sizing, and L2 mathematical normalization):

```bash
cargo test
```
