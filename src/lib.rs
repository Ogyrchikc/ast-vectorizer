//! AST Vectorizer Module
//!
//! This module provides a flexible interface to load HuggingFace models via ONNX Runtime
//! and perform text embedding (vectorization). It supports CPU, GPU, and NPU hardware acceleration.

use ort::ep::{CUDA, CoreML, DirectML, OpenVINO};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use thiserror::Error;

/// Errors that can occur during the vectorization process.
#[derive(Error, Debug)]
pub enum VectorizerError {
    #[error("Failed to load tokenizer: {0}")]
    TokenizerError(String),
    #[error("Failed to initialize ONNX session: {0}")]
    SessionError(String),
    #[error("Inference failed: {0}")]
    InferenceError(String),
    #[error("Tensor error: {0}")]
    TensorError(String),
}

/// Defines how token embeddings should be aggregated into a single sequence embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum PoolingStrategy {
    /// Average all token embeddings (ignoring padding tokens). Usually best for Sentence Transformers.
    Mean,
    /// Use only the first token ([CLS] token) embedding.
    Cls,
    /// Do not pool, just return the first token (fallback).
    None,
}

/// Target hardware device for ONNX execution.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum Device {
    /// Tries NPU, then GPU, and falls back to CPU automatically.
    Auto,
    /// Forces CPU execution.
    CPU,
    /// Prefers GPU execution (CUDA).
    GPU,
    /// Prefers NPU execution (CoreML, OpenVINO, DirectML).
    NPU,
}

/// Configuration for the Vectorizer.
pub struct VectorizerConfig {
    /// Path to the ONNX model file (.onnx).
    pub model_path: String,
    /// Path to the Tokenizer JSON file (tokenizer.json).
    pub tokenizer_path: String,
    /// Pooling strategy to apply to the output.
    pub pooling: PoolingStrategy,
    /// Preferred hardware device for inference.
    pub device: Device,
    /// Whether to pass token_type_ids to the model. True for models like MiniLM/BERT, false for RoBERTa/DistilBERT.
    pub use_token_type_ids: bool,
}

/// The core Vectorizer holding the ONNX session and the tokenizer.
pub struct Vectorizer {
    session: Session,
    tokenizer: Tokenizer,
    config: VectorizerConfig,
}

impl Vectorizer {
    /// Initializes a new Vectorizer instance.
    ///
    /// # Arguments
    /// * `config` - The configuration defining model paths and execution settings.
    pub fn new(config: VectorizerConfig) -> Result<Self, VectorizerError> {
        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| VectorizerError::TokenizerError(e.to_string()))?;

        let mut builder = Session::builder()
            .map_err(|e| VectorizerError::SessionError(e.to_string()))?;

        // Configure Hardware Acceleration (Execution Providers)
        builder = match config.device {
            Device::CPU => builder,
            Device::GPU => {
                builder.with_execution_providers([CUDA::default().build()])
                    .map_err(|e| VectorizerError::SessionError(e.to_string()))?
            },
            Device::NPU => {
                builder.with_execution_providers([
                    OpenVINO::default().build(),
                    CoreML::default().build(),
                    DirectML::default().build()
                ]).map_err(|e| VectorizerError::SessionError(e.to_string()))?
            },
            Device::Auto => {
                builder.with_execution_providers([
                    OpenVINO::default().build(),
                    CoreML::default().build(),
                    DirectML::default().build(),
                    CUDA::default().build(),
                ]).map_err(|e| VectorizerError::SessionError(e.to_string()))?
            }
        };

        let session = builder.commit_from_file(&config.model_path)
            .map_err(|e| VectorizerError::SessionError(e.to_string()))?;

        Ok(Self { session, tokenizer, config })
    }

    /// Embeds the given text into a normalized float vector.
    ///
    /// # Arguments
    /// * `text` - The input string (e.g., source code or a function representation).
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, VectorizerError> {
        // Step 1: Tokenization
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| VectorizerError::TokenizerError(e.to_string()))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        
        let seq_len = input_ids.len();

        // Step 2: Prepare Tensors
        let input_ids_array = ndarray::Array2::from_shape_vec((1, seq_len), input_ids.iter().map(|&x| x as i64).collect())
            .map_err(|e| VectorizerError::TensorError(e.to_string()))?;
        let attention_mask_array = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.iter().map(|&x| x as i64).collect())
            .map_err(|e| VectorizerError::TensorError(e.to_string()))?;

        let input_ids_tensor = Tensor::from_array(input_ids_array).map_err(|e| VectorizerError::TensorError(e.to_string()))?;
        let attention_mask_tensor = Tensor::from_array(attention_mask_array).map_err(|e| VectorizerError::TensorError(e.to_string()))?;

        // Step 3: Run Inference (Forward pass)
        let outputs = if self.config.use_token_type_ids {
            let type_ids_array = ndarray::Array2::from_shape_vec((1, seq_len), type_ids.iter().map(|&x| x as i64).collect())
                .map_err(|e| VectorizerError::TensorError(e.to_string()))?;
            let type_ids_tensor = Tensor::from_array(type_ids_array).map_err(|e| VectorizerError::TensorError(e.to_string()))?;
            
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => type_ids_tensor,
            ];
            self.session.run(inputs).map_err(|e| VectorizerError::InferenceError(e.to_string()))?
        } else {
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ];
            self.session.run(inputs).map_err(|e| VectorizerError::InferenceError(e.to_string()))?
        };

        // Extract raw data from the ONNX output
        let (output_shape, output_data) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| VectorizerError::TensorError(e.to_string()))?;

        let hidden_size = output_shape[2] as usize;
        let mut pooled = vec![0.0f32; hidden_size];

        // Step 4: Pooling
        match self.config.pooling {
            PoolingStrategy::Mean => {
                let mut sum_mask = 0.0;
                for i in 0..seq_len {
                    let mask = attention_mask[i] as f32;
                    // Ignore padding tokens
                    if mask > 0.0 {
                        for j in 0..hidden_size {
                            pooled[j] += output_data[i * hidden_size + j] * mask;
                        }
                        sum_mask += mask;
                    }
                }
                // Average the embeddings
                if sum_mask > 0.0 {
                    for j in 0..hidden_size {
                        pooled[j] /= sum_mask;
                    }
                }
            },
            PoolingStrategy::Cls => {
                for j in 0..hidden_size {
                    pooled[j] = output_data[0 * hidden_size + j];
                }
            },
            PoolingStrategy::None => {
                for j in 0..hidden_size {
                    pooled[j] = output_data[j];
                }
            }
        }

        // Step 5: L2 Normalization (Cosine Similarity preparation)
        let mut norm = 0.0;
        for j in 0..hidden_size {
            norm += pooled[j] * pooled[j];
        }
        
        norm = norm.sqrt();
        if norm > 0.0 {
            for j in 0..hidden_size {
                pooled[j] /= norm;
            }
        }

        Ok(pooled)
    }

    /// Returns a human-readable string indicating the chosen execution provider.
    pub fn device_info(&self) -> String {
        match self.config.device {
            Device::Auto => "Auto (NPU -> GPU -> CPU)".to_string(),
            Device::CPU => "CPU (Fallback)".to_string(),
            Device::GPU => "GPU (CUDA)".to_string(),
            Device::NPU => "NPU (CoreML / OpenVINO / DirectML)".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to get a working configuration for tests
    fn get_test_config() -> VectorizerConfig {
        VectorizerConfig {
            model_path: "models/all-MiniLM-L6-v2/model.onnx".to_string(),
            tokenizer_path: "models/all-MiniLM-L6-v2/tokenizer.json".to_string(),
            pooling: PoolingStrategy::Mean,
            device: Device::CPU, // Forced to CPU for reliable testing
            use_token_type_ids: true,
        }
    }

    #[test]
    fn test_initialization_fails_on_bad_path() {
        let config = VectorizerConfig {
            model_path: "invalid_path.onnx".to_string(),
            tokenizer_path: "models/all-MiniLM-L6-v2/tokenizer.json".to_string(),
            pooling: PoolingStrategy::Mean,
            device: Device::CPU,
            use_token_type_ids: true,
        };
        
        let result = Vectorizer::new(config);
        assert!(result.is_err(), "Vectorizer should fail to initialize with a bad model path");
    }

    #[test]
    fn test_successful_vectorization() {
        let mut vectorizer = Vectorizer::new(get_test_config()).expect("Failed to initialize test vectorizer");
        
        let sample_code = "def parse_ast(node):\n    pass";
        let vector = vectorizer.embed(sample_code).expect("Failed to embed text");
        
        // all-MiniLM-L6-v2 outputs exactly 384 dimensions
        assert_eq!(vector.len(), 384, "Embedding length must be exactly 384");
    }

    #[test]
    fn test_l2_normalization() {
        let mut vectorizer = Vectorizer::new(get_test_config()).expect("Failed to initialize test vectorizer");
        
        let vector = vectorizer.embed("Test sentence for normalization").expect("Failed to embed");
        
        // Calculate the sum of squares
        let mut sum_of_squares = 0.0;
        for &val in &vector {
            sum_of_squares += val * val;
        }
        
        // It should be extremely close to 1.0
        let difference = (1.0 - sum_of_squares).abs();
        assert!(difference < 1e-5, "Vector is not properly L2 normalized! Sum of squares: {}", sum_of_squares);
    }
}
