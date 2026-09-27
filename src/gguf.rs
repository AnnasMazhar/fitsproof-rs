//! Minimal GGUF header reader.
//!
//! Reads enough of a GGUF file to extract model architecture parameters
//! (`ModelConfig`) without loading the tensor data.  This is sufficient for
//! `plan()` to predict peak memory usage for a real model.
//!
//! # GGUF format reference
//!
//! GGUF format versions 1, 2, and 3: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//!
//! Header layout:
//!   - magic: u32 = 0x46554747 ("GGUF")
//!   - version: u32 (1, 2, or 3)
//!   - tensor_count: u64
//!   - metadata_kv_count: u64
//!   - metadata_kv: repeated (key: string, value_type: u32, value)
//!
//! String format: (len: u64, bytes: [u8; len])

use std::io::{self, Read};

/// Errors from GGUF header reading.
#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    MissingField(String),
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "IO error: {e}"),
            GgufError::InvalidMagic(m) => write!(f, "invalid GGUF magic: 0x{m:08x}"),
            GgufError::UnsupportedVersion(v) => write!(f, "unsupported GGUF version: {v}"),
            GgufError::MissingField(k) => write!(f, "missing required field: {k}"),
        }
    }
}

impl From<io::Error> for GgufError {
    fn from(e: io::Error) -> Self {
        GgufError::Io(e)
    }
}

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" stored as bytes [0x47,0x47,0x55,0x46], read LE = 0x46554747

/// Key-value metadata from the GGUF header.
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv: std::collections::HashMap<String, GgufValue>,
}

/// A GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::U64(v) => Some(*v as u32),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U8(v) => Some(*v as u32),
            GgufValue::U16(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u32().map(|v| v as usize)
    }
}

struct Reader<R: Read> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn new(inner: R) -> Self {
        Self { inner }
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        let mut buf = [0u8; 1];
        self.inner.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let mut buf = [0u8; 2];
        self.inner.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        self.inner.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let mut buf = [0u8; 8];
        self.inner.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_u32()?.to_le_bytes()))
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.read_u64()?.to_le_bytes()))
    }

    fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        // Cap at 4 KB to avoid ridiculous allocations on corrupt files.
        if len > 4096 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GGUF string too long: {len} bytes"),
            ));
        }
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf)?;
        String::from_utf8(buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    }

    fn read_value(&mut self, value_type: u32) -> io::Result<GgufValue> {
        match value_type {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_u8()? as i8)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_u16()? as i16)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                // Array: element_type (u32) + count (u64) + elements
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                // Cap array size to avoid OOM on corrupt files.
                let cap = count.min(256);
                let mut arr = Vec::with_capacity(cap);
                for _ in 0..count {
                    arr.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown GGUF value type: {other}"),
            )),
        }
    }
}

/// Read GGUF metadata from a reader (file, cursor, etc.).
///
/// Only reads the header and key-value metadata; does not load tensor data.
pub fn read_metadata<R: Read>(reader: R) -> Result<GgufMetadata, GgufError> {
    let mut r = Reader::new(reader);

    // Magic
    let magic = r.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(magic));
    }

    // Version
    let version = r.read_u32()?;
    if version == 0 || version > 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    // Tensor count + metadata count
    let tensor_count = r.read_u64()?;
    let kv_count = r.read_u64()?;

    // Read key-value metadata pairs.
    let mut kv = std::collections::HashMap::new();
    for _ in 0..kv_count {
        let key = match r.read_string() {
            Ok(k) => k,
            Err(_) => break, // stop on malformed entry
        };
        let value_type = r.read_u32()?;
        let value = match r.read_value(value_type) {
            Ok(v) => v,
            Err(_) => break,
        };
        kv.insert(key, value);
    }

    Ok(GgufMetadata {
        version,
        tensor_count,
        kv,
    })
}

/// Extract a `ModelConfig` from GGUF metadata.
///
/// Uses standard GGUF keys (e.g. `llama.context_length`, `llama.block_count`, etc.).
/// Falls back to defaults for fields not present in the metadata.
pub fn metadata_to_model_config(
    meta: &GgufMetadata,
    model_name: &str,
) -> Result<crate::model::ModelConfig, GgufError> {
    // Try common architecture prefixes (llama, qwen2, gemma, phi, mistral, etc.)
    let prefixes = [
        "llama", "qwen2", "qwen3", "gemma", "phi", "mistral", "falcon", "gpt2", "bloom",
    ];

    // Helper: look up a key with multiple prefixes.
    let get = |suffix: &str| -> Option<&GgufValue> {
        for prefix in &prefixes {
            if let Some(v) = meta.kv.get(&format!("{prefix}.{suffix}")) {
                return Some(v);
            }
        }
        meta.kv.get(suffix)
    };

    let num_layers = get("block_count")
        .and_then(|v| v.as_usize())
        .ok_or_else(|| GgufError::MissingField("*.block_count".into()))?;

    let hidden_size = get("embedding_length")
        .and_then(|v| v.as_usize())
        .ok_or_else(|| GgufError::MissingField("*.embedding_length".into()))?;

    let num_heads = get("attention.head_count")
        .and_then(|v| v.as_usize())
        .ok_or_else(|| GgufError::MissingField("*.attention.head_count".into()))?;

    let num_kv_heads = get("attention.head_count_kv")
        .and_then(|v| v.as_usize())
        .unwrap_or(num_heads); // default: MHA (num_kv_heads = num_heads)

    let head_dim = hidden_size.checked_div(num_heads).unwrap_or(64);

    let intermediate_size = get("feed_forward_length")
        .and_then(|v| v.as_usize())
        .unwrap_or(hidden_size * 4); // default: 4× hidden

    let vocab_size = get("vocab_size")
        .and_then(|v| v.as_usize())
        .or_else(|| {
            meta.kv.get("tokenizer.ggml.tokens").and_then(|v| {
                if let GgufValue::Array(arr) = v {
                    Some(arr.len())
                } else {
                    None
                }
            })
        })
        .unwrap_or(32000); // default: typical small-model vocab

    let max_seq_len = get("context_length")
        .and_then(|v| v.as_usize())
        .unwrap_or(4096);

    Ok(crate::model::ModelConfig {
        num_layers,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        intermediate_size,
        vocab_size,
        max_seq_len,
        name: model_name.to_string(),
    })
}

#[cfg(test)]
mod tests {
    //! Tests for the GGUF reader.

    use super::*;
    use std::io::Cursor;

    /// Construct a minimal valid GGUF v2 header with one uint32 KV entry.
    fn make_minimal_gguf_v2() -> Vec<u8> {
        let mut buf = Vec::new();
        // magic = 0x46554747 (bytes: G G U F read as LE u32)
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // version = 2
        buf.extend_from_slice(&2u32.to_le_bytes());
        // tensor_count = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        // kv_count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // key = "test.key" (8 bytes)
        let key = b"test.key";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        // value_type = 4 (u32)
        buf.extend_from_slice(&4u32.to_le_bytes());
        // value = 42
        buf.extend_from_slice(&42u32.to_le_bytes());
        buf
    }

    /// Fault detected: magic check accepts any 4 bytes.
    #[test]
    fn invalid_magic_returns_error() {
        let data = vec![0xAA, 0xBB, 0xCC, 0xDD, 0, 0, 0, 0]; // wrong magic
        let result = read_metadata(Cursor::new(data));
        assert!(
            matches!(result, Err(GgufError::InvalidMagic(_))),
            "wrong magic must return GgufError::InvalidMagic"
        );
    }

    /// Fault detected: valid GGUF header rejected (magic parsing wrong).
    #[test]
    fn valid_gguf_v2_parses_ok() {
        let data = make_minimal_gguf_v2();
        let meta = read_metadata(Cursor::new(data)).expect("valid GGUF v2 must parse");
        assert_eq!(meta.version, 2);
        assert_eq!(meta.tensor_count, 0);
        assert!(meta.kv.contains_key("test.key"), "KV must contain test.key");
    }

    /// Fault detected: KV value parsed as wrong type (u32 read as u64).
    #[test]
    fn kv_u32_value_reads_correctly() {
        let data = make_minimal_gguf_v2();
        let meta = read_metadata(Cursor::new(data)).unwrap();
        let val = meta.kv.get("test.key").unwrap();
        assert_eq!(val.as_u32(), Some(42), "KV u32 value must be 42");
    }

    /// Fault detected: metadata_to_model_config returns garbage when fields are missing.
    #[test]
    fn metadata_to_config_fails_on_missing_required_fields() {
        let meta = GgufMetadata {
            version: 2,
            tensor_count: 0,
            kv: std::collections::HashMap::new(),
        };
        let result = metadata_to_model_config(&meta, "test");
        assert!(
            result.is_err(),
            "must return error when required fields are missing"
        );
    }
}
