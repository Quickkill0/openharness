use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncSeekExt, AsyncWriteExt, BufReader, SeekFrom};
use crate::error::AppError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCounts {
    pub prompt: u32,
    pub completion: u32,
}

/// One plain chat entry in the JSONL transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRecord {
    pub seq: u64,
    /// "user" | "assistant" | "system"
    pub role: String,
    pub content: String,
    pub ts: i64,                // unix epoch seconds
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tokens: Option<TokenCounts>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub finish_reason: Option<String>,  // "stop" | "length" | "cancelled" | ...
}

#[derive(Debug, Clone, Copy)]
pub struct MessageMeta {
    pub seq: u64,
    pub byte_offset: u64,
    pub byte_len: u64,
}

pub struct TranscriptWriter {
    file: File,
    pub path: PathBuf,
    next_seq: u64,
    byte_cursor: u64,
}

impl TranscriptWriter {
    /// Open existing transcript or create a new one. Scans existing file to recover
    /// `next_seq` and `byte_cursor`.
    pub async fn open_or_create(path: PathBuf) -> Result<Self, AppError> {
        // Ensure parent dir
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| AppError::TranscriptIoError(format!("create dir: {}", e)))?;
        }

        let (next_seq, byte_cursor) = if path.exists() {
            // Scan existing
            let f = File::open(&path).await.map_err(|e| AppError::TranscriptIoError(format!("open scan: {}", e)))?;
            let mut reader = BufReader::new(f);
            let mut buf = String::new();
            let mut seq = 0u64;
            let mut offset = 0u64;
            loop {
                buf.clear();
                let n = reader.read_line(&mut buf).await.map_err(|e| AppError::TranscriptIoError(format!("read line: {}", e)))?;
                if n == 0 { break; }
                // Parse to recover the max seq
                if let Ok(rec) = serde_json::from_str::<MessageRecord>(buf.trim_end_matches('\n')) {
                    if rec.seq + 1 > seq { seq = rec.seq + 1; }
                }
                offset += n as u64;
            }
            (seq, offset)
        } else { (0, 0) };

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|e| AppError::TranscriptIoError(format!("open append: {}", e)))?;

        Ok(Self { file, path, next_seq, byte_cursor })
    }

    /// Append a chat entry.
    pub async fn append(
        &mut self,
        role: &str,
        content: &str,
        ts: i64,
        tokens: Option<TokenCounts>,
        finish_reason: Option<String>,
    ) -> Result<(MessageMeta, MessageRecord), AppError> {
        let record = MessageRecord {
            seq: self.next_seq,
            role: role.to_string(),
            content: content.to_string(),
            ts,
            tokens,
            finish_reason,
        };
        let mut line = serde_json::to_string(&record).map_err(|e| AppError::TranscriptIoError(format!("serialize: {}", e)))?;
        line.push('\n');
        let bytes = line.as_bytes();
        let byte_offset = self.byte_cursor;
        let byte_len = bytes.len() as u64;
        self.file.write_all(bytes).await.map_err(|e| AppError::TranscriptIoError(format!("write: {}", e)))?;
        self.file.flush().await.map_err(|e| AppError::TranscriptIoError(format!("flush: {}", e)))?;
        let meta = MessageMeta { seq: self.next_seq, byte_offset, byte_len };
        self.next_seq += 1;
        self.byte_cursor += byte_len;
        Ok((meta, record))
    }
}

pub async fn read_messages(path: &Path) -> Result<Vec<MessageRecord>, AppError> {
    if !path.exists() { return Ok(Vec::new()); }
    let f = File::open(path).await.map_err(|e| AppError::TranscriptIoError(format!("open: {}", e)))?;
    let mut reader = BufReader::new(f);
    let mut out = Vec::new();
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf).await.map_err(|e| AppError::TranscriptIoError(format!("read: {}", e)))?;
        if n == 0 { break; }
        let line = buf.trim_end_matches('\n');
        if line.is_empty() { continue; }
        let rec: MessageRecord = serde_json::from_str(line).map_err(|e| AppError::TranscriptIoError(format!("parse line: {}", e)))?;
        out.push(rec);
    }
    Ok(out)
}

pub async fn read_message_at(path: &Path, offset: u64, len: u64) -> Result<MessageRecord, AppError> {
    let mut f = File::open(path).await.map_err(|e| AppError::TranscriptIoError(format!("open: {}", e)))?;
    f.seek(SeekFrom::Start(offset)).await.map_err(|e| AppError::TranscriptIoError(format!("seek: {}", e)))?;
    use tokio::io::AsyncReadExt;
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).await.map_err(|e| AppError::TranscriptIoError(format!("read: {}", e)))?;
    let s = std::str::from_utf8(&buf).map_err(|e| AppError::TranscriptIoError(format!("utf8: {}", e)))?.trim_end_matches('\n');
    let rec: MessageRecord = serde_json::from_str(s).map_err(|e| AppError::TranscriptIoError(format!("parse: {}", e)))?;
    Ok(rec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn round_trip_basic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c1.jsonl");
        let mut w = TranscriptWriter::open_or_create(path.clone()).await.unwrap();
        let (m1, _) = w.append("user", "hello", 1000, None, None).await.unwrap();
        let (m2, _) = w.append("assistant", "world", 1001, Some(TokenCounts { prompt: 3, completion: 1 }), Some("stop".into())).await.unwrap();
        assert_eq!(m1.seq, 0);
        assert_eq!(m2.seq, 1);
        let all = read_messages(&path).await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].content, "hello");
        assert_eq!(all[1].content, "world");
        // Byte-slice round-trip
        let rec = read_message_at(&path, m2.byte_offset, m2.byte_len).await.unwrap();
        assert_eq!(rec.content, "world");
        assert_eq!(rec.tokens.unwrap().completion, 1);
    }

    #[test]
    fn legacy_chat_line_parses() {
        // Older chat transcripts without optional fields still deserialize.
        let line = r#"{"seq":0,"role":"user","content":"hi","ts":100}"#;
        let rec: MessageRecord = serde_json::from_str(line).unwrap();
        assert_eq!(rec.role, "user");
        assert!(rec.tokens.is_none() && rec.finish_reason.is_none());
    }

    #[tokio::test]
    async fn reopen_recovers_seq() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("c2.jsonl");
        {
            let mut w = TranscriptWriter::open_or_create(path.clone()).await.unwrap();
            w.append("user", "a", 1, None, None).await.unwrap();
            w.append("assistant", "b", 2, None, None).await.unwrap();
        }
        let mut w = TranscriptWriter::open_or_create(path.clone()).await.unwrap();
        let (m, _) = w.append("user", "c", 3, None, None).await.unwrap();
        assert_eq!(m.seq, 2);
        let all = read_messages(&path).await.unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[2].content, "c");
    }
}
