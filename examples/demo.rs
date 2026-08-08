use ast_vectorizer::{Vectorizer, VectorizerConfig, Device, PoolingStrategy};

fn main() {
    let config = VectorizerConfig {
        model_path: "models/all-MiniLM-L6-v2/model.onnx".to_string(),
        tokenizer_path: "models/all-MiniLM-L6-v2/tokenizer.json".to_string(),
        pooling: PoolingStrategy::Mean,
        device: Device::Auto,
    };

    println!("Loading model...");
    let mut vectorizer = match Vectorizer::new(config) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Failed to initialize vectorizer: {}", e);
            return;
        }
    };

    println!("Model loaded successfully!");
    println!("Hardware Configuration: {}", vectorizer.device_info());

    let sample_text = "def calculate_sum(a, b):\n    return a + b";
    println!("\nVectorizing sample text...");
    
    match vectorizer.embed(sample_text) {
        Ok(vector) => {
            println!("Success! Vector length: {}", vector.len());
            println!("First 5 elements: {:?}", &vector[..5]);
        },
        Err(e) => eprintln!("Error during vectorization: {}", e),
    }
}
