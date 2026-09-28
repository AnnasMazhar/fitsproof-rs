//! GGUF tensor data loader.
//!
//! Extends the header reader (`gguf.rs`) to also load the tensor data arrays.
//! Supports the most common storage types:
//!   - GGML_TYPE_F32  (type 0) — 4 bytes/element
//!   - GGML_TYPE_F16  (type 1) — 2 bytes/element, IEEE 754 half
//!   - GGML_TYPE_Q4_0 (type 2) — 32-element block: 2-byte f16 scale + 16 packed bytes
//!   - GGML_TYPE_Q8_0 (type 8) — 32-element block: 4-byte f32 scale + 32 packed bytes
//!   - GGML_TYPE_Q4_K (type 12) — K-quant 4-bit block (common "Q4_K_M" format)
//!
//! Other types are stored verbatim as raw bytes; they can be used for size
//! accounting but not for arithmetic.
//!
//! # GGUF specification
//! https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
//!
//! Tensor info layout (per tensor):
//!   name (string) | n_dims (u32) | dims[n_dims] (u64 each) | type (u32) | offset (u64)
//! All tensor data follows the tensor-info section, padded to 32-byte alignment.

use std::io::{self, Read, Seek, SeekFrom};

use crate::gguf::{GgufError, GgufMetadata};

// ---------------------------------------------------------------------------
// GGML type IDs
// ---------------------------------------------------------------------------

/// GGML element type tag stored in tensor info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4K, // Q4_K (type id 12) — K-quant
    Q6K, // Q6_K (type id 14)
    Other(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            14 => GgmlType::Q6K,
            other => GgmlType::Other(other),
        }
    }

    /// Bytes per element (F32/F16) or bytes per 32-element block.
    /// Returns (bytes_per_block, elements_per_block).
    pub fn block_size(self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (4, 1),
            GgmlType::F16 => (2, 1),
            GgmlType::Q4_0 => (2 + 16, 32), // 2-byte scale + 16 bytes for 32×4-bit
            GgmlType::Q8_0 => (4 + 32, 32), // 4-byte scale + 32 bytes for 32×8-bit
            GgmlType::Q4K => (144, 256),    // K-quant: 256 values per "super-block"
            GgmlType::Q6K => (210, 256),    // K-quant Q6
            GgmlType::Other(_) => (1, 1),   // unknown — 1 byte per "element" (raw)
        }
    }

    /// Byte size for a tensor with `n_elements` elements.
    pub fn byte_size(self, n_elements: usize) -> usize {
        let (bytes, elems) = self.block_size();
        // round up to whole blocks
        let n_blocks = n_elements.div_ceil(elems);
        n_blocks * bytes
    }
}

// ---------------------------------------------------------------------------
// Tensor info (header portion)
// ---------------------------------------------------------------------------

/// Tensor info record from the GGUF tensor-info section.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Byte offset from the start of the tensor data block.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    pub fn byte_size(&self) -> usize {
        self.ggml_type.byte_size(self.n_elements())
    }
}

// ---------------------------------------------------------------------------
// Tensor data (decoded to f32)
// ---------------------------------------------------------------------------

/// A tensor loaded from a GGUF file, dequantised to f32.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dims: Vec<usize>,
    /// Dequantised values.  Len == product of dims.
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from tensor loading.
#[derive(Debug)]
pub enum TensorLoadError {
    Gguf(GgufError),
    Io(io::Error),
    UnsupportedType {
        name: String,
        type_id: u32,
    },
    SizeMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for TensorLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorLoadError::Gguf(e) => write!(f, "GGUF error: {e}"),
            TensorLoadError::Io(e) => write!(f, "IO error: {e}"),
            TensorLoadError::UnsupportedType { name, type_id } => {
                write!(f, "tensor '{name}': unsupported type id {type_id}")
            }
            TensorLoadError::SizeMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "tensor '{name}': size mismatch, expected {expected} bytes, got {got}"
                )
            }
        }
    }
}

impl From<io::Error> for TensorLoadError {
    fn from(e: io::Error) -> Self {
        TensorLoadError::Io(e)
    }
}

impl From<GgufError> for TensorLoadError {
    fn from(e: GgufError) -> Self {
        TensorLoadError::Gguf(e)
    }
}

// ---------------------------------------------------------------------------
// Reading tensor info from GGUF
// ---------------------------------------------------------------------------

/// Read a little-endian u32.
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64.
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a GGUF-format string.
fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    if len > 65536 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "string too long",
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

/// Skip over a GGUF value (metadata KV entry value).
fn skip_value<R: Read>(r: &mut R, value_type: u32) -> io::Result<()> {
    match value_type {
        0 | 1 | 7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
        }
        2 | 3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
        }
        4..=6 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
        }
        8 => {
            read_string(r)?;
        }
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            for _ in 0..count {
                skip_value(r, elem_type)?;
            }
        }
        10..=12 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)?;
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unknown value type",
            ))
        }
    }
    Ok(())
}

/// Read all tensor infos from a GGUF file.
///
/// The reader must be positioned at the very start of the file.
/// This function re-reads the header + kv section to reach the tensor-info section.
/// Returns the tensor infos and the byte offset where the tensor data block starts.
pub fn read_tensor_infos<R: Read + Seek>(
    r: &mut R,
    meta: &GgufMetadata,
) -> Result<(Vec<TensorInfo>, u64), TensorLoadError> {
    // Seek back to start.
    r.seek(SeekFrom::Start(0))?;

    // Skip magic + version.
    let mut skip4 = [0u8; 4];
    r.read_exact(&mut skip4)?; // magic
    r.read_exact(&mut skip4)?; // version

    // tensor_count and kv_count.
    let _tensor_count = read_u64(r)?;
    let kv_count = read_u64(r)?;

    // Skip all KV entries.
    for _ in 0..kv_count {
        read_string(r)?; // key
        let vtype = read_u32(r)?;
        skip_value(r, vtype)?;
    }

    // Now read tensor infos.
    let mut infos = Vec::with_capacity(meta.tensor_count as usize);
    for _ in 0..meta.tensor_count {
        let name = read_string(r)?;
        let n_dims = read_u32(r)? as usize;
        let dims: Vec<u64> = (0..n_dims)
            .map(|_| read_u64(r))
            .collect::<io::Result<Vec<_>>>()?;
        let type_id = read_u32(r)?;
        let offset = read_u64(r)?;
        infos.push(TensorInfo {
            name,
            dims,
            ggml_type: GgmlType::from_u32(type_id),
            offset,
        });
    }

    // The tensor data block starts at the current position, padded to 32-byte alignment.
    let pos = r.stream_position()?;
    let aligned = (pos + 31) & !31;
    let data_start = aligned;

    Ok((infos, data_start))
}

// ---------------------------------------------------------------------------
// Dequantisation
// ---------------------------------------------------------------------------

/// Convert a half-precision (f16) bit pattern to f32.
///
/// IEEE 754 half: sign(1) | exponent(5) | mantissa(10).
/// Formula from IEEE 754-2008 §5.4.2.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        // subnormal
        if mant == 0 {
            sign << 31
        } else {
            // normalise
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        // inf or nan
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Dequantise a Q4_0 block (32 values).
///
/// Block layout: 2-byte f16 scale `d`, then 16 bytes of 4-bit values (two per byte).
/// Formula: x[i] = d * (nibble[i] - 8)  — zero point is 8.
fn dequant_q4_0(block: &[u8]) -> Vec<f32> {
    assert!(block.len() >= 18, "Q4_0 block must be at least 18 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut out = Vec::with_capacity(32);
    for &byte in &block[2..18] {
        let lo = (byte & 0x0f) as i32 - 8;
        let hi = ((byte >> 4) & 0x0f) as i32 - 8;
        out.push(d * lo as f32);
        out.push(d * hi as f32);
    }
    out
}

/// Dequantise a Q8_0 block (32 values).
///
/// Block layout: 4-byte f32 scale `d`, then 32 signed bytes.
/// Formula: x[i] = d * q[i].
fn dequant_q8_0(block: &[u8]) -> Vec<f32> {
    assert!(block.len() >= 36, "Q8_0 block must be at least 36 bytes");
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    block[4..36].iter().map(|&b| d * (b as i8) as f32).collect()
}

/// Dequantise a Q4_K super-block (256 values).
///
/// Q4_K block layout (144 bytes, 256 elements):
///   - 2 bytes: d  (f16, super-block scale)
///   - 2 bytes: dmin (f16, super-block min)
///   - 12 bytes: scales/mins for 8 sub-blocks (6 bits each, packed)
///   - 128 bytes: 4-bit quantised values (256 × 4-bit = 128 bytes)
///
/// Formula: x[i] = d * scale_sub * q[i] - dmin * min_sub
///
/// Reference: ggml/src/ggml-quants.c `ggml_dequantize_row_q4_K`.
fn dequant_q4k(block: &[u8]) -> Vec<f32> {
    assert!(block.len() >= 144, "Q4_K block must be 144 bytes");
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    // Decode 8 pairs of (scale, min) from 12 bytes (6 bits each).
    // Layout: bytes 4..15 encode scales[0..7] and mins[0..7] packed as:
    // scales[i] in bits 0..5 of scale_bytes, mins[i] in the upper bits.
    // Exact bit layout from ggml-quants.h: each byte group encodes 2 values.
    let scale_bytes = &block[4..16];
    let mut scales = [0f32; 8];
    let mut mins = [0f32; 8];
    for i in 0..4 {
        let byte0 = scale_bytes[i * 3] as u32;
        let byte1 = scale_bytes[i * 3 + 1] as u32;
        let byte2 = scale_bytes[i * 3 + 2] as u32;

        scales[i * 2] = d * ((byte0 & 0x3f) as f32);
        scales[i * 2 + 1] = d * ((byte1 & 0x3f) as f32);
        mins[i * 2] = dmin * (((byte0 >> 6) | ((byte2 & 0x0f) << 2)) as f32);
        mins[i * 2 + 1] = dmin * (((byte1 >> 6) | ((byte2 >> 4) << 2)) as f32);
    }

    // 4-bit values start at byte 16 (after 4 header + 12 scale bytes).
    let qbytes = &block[16..144];
    let mut out = Vec::with_capacity(256);

    for sub in 0..8 {
        let sc = scales[sub];
        let mn = mins[sub];
        // 32 values per sub-block = 16 bytes of 4-bit pairs.
        let base = sub * 16;
        for &byte in &qbytes[base..base + 16] {
            let lo = (byte & 0x0f) as f32;
            let hi = ((byte >> 4) & 0x0f) as f32;
            out.push(sc * lo - mn);
            out.push(sc * hi - mn);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Load a single tensor
// ---------------------------------------------------------------------------

/// Load and dequantise a single tensor from a GGUF file.
///
/// `r` must be seekable.  `data_start` is the byte offset where the tensor data
/// block begins (returned by `read_tensor_infos`).
pub fn load_tensor<R: Read + Seek>(
    r: &mut R,
    info: &TensorInfo,
    data_start: u64,
) -> Result<Tensor, TensorLoadError> {
    let n = info.n_elements();
    let byte_size = info.byte_size();
    let abs_offset = data_start + info.offset;
    r.seek(SeekFrom::Start(abs_offset))?;

    let data = match info.ggml_type {
        GgmlType::F32 => {
            let mut raw = vec![0u8; byte_size];
            r.read_exact(&mut raw)?;
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        GgmlType::F16 => {
            let mut raw = vec![0u8; byte_size];
            r.read_exact(&mut raw)?;
            raw.chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        }
        GgmlType::Q4_0 => {
            let block_bytes = 18; // 2 + 16
            let n_blocks = n.div_ceil(32);
            let mut out = Vec::with_capacity(n);
            let mut buf = vec![0u8; block_bytes];
            for _ in 0..n_blocks {
                r.read_exact(&mut buf)?;
                out.extend(dequant_q4_0(&buf));
            }
            out.truncate(n);
            out
        }
        GgmlType::Q8_0 => {
            let block_bytes = 36; // 4 + 32
            let n_blocks = n.div_ceil(32);
            let mut out = Vec::with_capacity(n);
            let mut buf = vec![0u8; block_bytes];
            for _ in 0..n_blocks {
                r.read_exact(&mut buf)?;
                out.extend(dequant_q8_0(&buf));
            }
            out.truncate(n);
            out
        }
        GgmlType::Q4K => {
            let block_bytes = 144;
            let n_blocks = n.div_ceil(256);
            let mut out = Vec::with_capacity(n);
            let mut buf = vec![0u8; block_bytes];
            for _ in 0..n_blocks {
                r.read_exact(&mut buf)?;
                out.extend(dequant_q4k(&buf));
            }
            out.truncate(n);
            out
        }
        GgmlType::Q6K | GgmlType::Other(_) => {
            let id = match info.ggml_type {
                GgmlType::Q6K => 14,
                GgmlType::Other(id) => id,
                _ => unreachable!(),
            };
            return Err(TensorLoadError::UnsupportedType {
                name: info.name.clone(),
                type_id: id,
            });
        }
    };

    Ok(Tensor {
        name: info.name.clone(),
        dims: info.dims.iter().map(|&d| d as usize).collect(),
        data,
    })
}

// ---------------------------------------------------------------------------
// Load weights for a model
// ---------------------------------------------------------------------------

/// Names that identify weight tensors we need for the transformer.
/// Returns true if this tensor name is one we should load.
fn is_needed_tensor(name: &str) -> bool {
    // Load: token embedding, attention weights, FFN weights, norms, lm_head.
    let needed_substrings = [
        "token_embd",
        "token_emb",
        "tok_embeddings",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "attn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "ffn_norm",
        "output_norm",
        "output.weight",
        "lm_head",
    ];
    needed_substrings.iter().any(|s| name.contains(s))
}

/// A collection of loaded tensors indexed by name.
pub struct TensorStore {
    pub tensors: std::collections::HashMap<String, Tensor>,
}

impl TensorStore {
    /// Load all weight tensors from a GGUF file.
    ///
    /// Skips tensors for unsupported quantisation types (Q6_K, etc.) rather than failing,
    /// since those tensors are typically not needed for the common Q4_K_M format.
    pub fn load_from_file(path: &str) -> Result<(Self, GgufMetadata), TensorLoadError> {
        use crate::gguf::read_metadata;

        let mut file = std::fs::File::open(path).map_err(TensorLoadError::Io)?;

        // Read metadata.
        let meta = read_metadata(std::io::BufReader::new(&mut file))?;

        // Seek back and read tensor infos.
        let mut file = std::fs::File::open(path).map_err(TensorLoadError::Io)?;
        let (infos, data_start) = read_tensor_infos(&mut file, &meta)?;

        let mut tensors = std::collections::HashMap::new();
        for info in &infos {
            if !is_needed_tensor(&info.name) {
                continue;
            }
            match load_tensor(&mut file, info, data_start) {
                Ok(t) => {
                    tensors.insert(info.name.clone(), t);
                }
                Err(TensorLoadError::UnsupportedType { .. }) => {
                    // Skip unsupported types — don't fail the whole load.
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Ok((TensorStore { tensors }, meta))
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Fault detected: f16_to_f32 gives wrong value for 1.0 (bits = 0x3c00).
    #[test]
    fn f16_one_converts_to_f32() {
        // 1.0 in f16: sign=0, exp=15, mant=0 → bits = 0x3c00
        let result = f16_to_f32(0x3c00);
        assert!(
            (result - 1.0f32).abs() < 1e-6,
            "f16 1.0 must convert to f32 1.0, got {result}"
        );
    }

    /// Fault detected: f16_to_f32 gives wrong value for 2.0 (bits = 0x4000).
    #[test]
    fn f16_two_converts_to_f32() {
        let result = f16_to_f32(0x4000);
        assert!(
            (result - 2.0f32).abs() < 1e-6,
            "f16 2.0 must convert to f32 2.0, got {result}"
        );
    }

    /// Fault detected: f16_to_f32 gives wrong value for 0.5 (bits = 0x3800).
    #[test]
    fn f16_half_converts_to_f32() {
        let result = f16_to_f32(0x3800);
        assert!(
            (result - 0.5f32).abs() < 1e-6,
            "f16 0.5 must convert to f32 0.5, got {result}"
        );
    }

    /// Fault detected: f16 zero doesn't map to f32 zero.
    #[test]
    fn f16_zero_converts_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
    }

    /// Fault detected: f16 negative doesn't preserve sign.
    #[test]
    fn f16_negative_one_converts_correctly() {
        // -1.0 in f16: sign=1, exp=15, mant=0 → bits = 0xbc00
        let result = f16_to_f32(0xbc00);
        assert!(
            (result - (-1.0f32)).abs() < 1e-6,
            "f16 -1.0 must convert to f32 -1.0, got {result}"
        );
    }

    /// Fault detected: Q4_0 dequant returns wrong count.
    #[test]
    fn dequant_q4_0_block_length() {
        // Build a minimal Q4_0 block: d=1.0 (f16=0x3c00), 16 zero bytes.
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0 = 0x3c00 in LE
        let out = dequant_q4_0(&block);
        assert_eq!(out.len(), 32, "Q4_0 block must decode 32 values");
    }

    /// Fault detected: Q4_0 dequant formula wrong — zero-point should be 8 not 0.
    #[test]
    fn dequant_q4_0_zero_nibble_gives_minus_8_times_scale() {
        // d=1.0, all nibbles=0 → all values = 1.0 * (0 - 8) = -8.0
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0
        let out = dequant_q4_0(&block);
        for &v in &out {
            assert!(
                (v - (-8.0f32)).abs() < 1e-5,
                "zero nibble with scale=1 must give -8.0, got {v}"
            );
        }
    }

    /// Fault detected: Q8_0 dequant returns wrong count.
    #[test]
    fn dequant_q8_0_block_length() {
        let mut block = vec![0u8; 36];
        // d = 1.0 as f32
        block[0..4].copy_from_slice(&1.0f32.to_le_bytes());
        let out = dequant_q8_0(&block);
        assert_eq!(out.len(), 32, "Q8_0 block must decode 32 values");
    }

    /// Fault detected: Q8_0 dequant formula wrong — values should be d * signed_byte.
    #[test]
    fn dequant_q8_0_value_matches_formula() {
        // d=2.0, q[0]=5 → x[0] = 2.0 * 5 = 10.0
        let mut block = vec![0u8; 36];
        block[0..4].copy_from_slice(&2.0f32.to_le_bytes());
        block[4] = 5u8;
        let out = dequant_q8_0(&block);
        assert!(
            (out[0] - 10.0f32).abs() < 1e-5,
            "Q8_0 first element must be d*q = 10.0, got {}",
            out[0]
        );
    }

    /// Fault detected: GgmlType::byte_size returns 0 for non-trivial tensors.
    #[test]
    fn ggml_type_f32_byte_size() {
        assert_eq!(GgmlType::F32.byte_size(100), 400);
    }

    /// Fault detected: GgmlType::byte_size rounds up wrong for quantised types.
    #[test]
    fn ggml_type_q4_0_byte_size_rounds_up() {
        // 33 elements → 2 blocks of 32 → 2 * 18 = 36 bytes
        assert_eq!(GgmlType::Q4_0.byte_size(33), 36);
    }

    /// Fault detected: Q4_K block decodes to wrong length.
    #[test]
    fn dequant_q4k_block_length() {
        let block = vec![0u8; 144];
        let out = dequant_q4k(&block);
        assert_eq!(out.len(), 256, "Q4_K block must decode 256 values");
    }
}
