use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::AppError;

#[derive(Debug, Clone)]
pub enum SseChunk {
    Delta(String),
    /// A `delta.reasoning_content` token. llama-server's OpenAI-compat layer
    /// splits "thinking" model output (e.g. Qwen3 `<think>` blocks) into this
    /// side-channel unless the request sets `reasoning_format: "none"`.
    /// Dropping these silently is how reasoning models used to produce empty
    /// turns — every consumer must route them somewhere deliberate.
    Reasoning(String),
    Usage { prompt: u32, completion: u32 },
    FinishReason(String),
    Done,
}

#[derive(Debug, Deserialize)]
struct DeltaPayload {
    #[serde(default)]
    content: Option<String>,
    /// llama-server emits the model's thinking here when `reasoning_format`
    /// is left at its default (`auto`/`deepseek`).
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChoiceChunk {
    #[serde(default)]
    delta: Option<DeltaPayload>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<ChoiceChunk>,
    #[serde(default)]
    usage: Option<UsageChunk>,
}

#[derive(Debug, Deserialize)]
struct UsageChunk {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// Parse one SSE `data:` payload into the chunks it carries, in stream order:
/// reasoning before content (llama-server emits thinking first), then the
/// finish reason, then usage. Extracted from the stream loop so the parsing
/// of real llama-server payloads is unit-testable.
fn parse_sse_data(data: &str) -> Result<Vec<SseChunk>, serde_json::Error> {
    let chunk: StreamChunk = serde_json::from_str(data)?;
    let mut out = Vec::new();
    if let Some(choice) = chunk.choices.first() {
        if let Some(delta) = &choice.delta {
            if let Some(reasoning) = &delta.reasoning_content {
                if !reasoning.is_empty() {
                    out.push(SseChunk::Reasoning(reasoning.clone()));
                }
            }
            if let Some(content) = &delta.content {
                if !content.is_empty() {
                    out.push(SseChunk::Delta(content.clone()));
                }
            }
        }
        if let Some(fr) = &choice.finish_reason {
            out.push(SseChunk::FinishReason(fr.clone()));
        }
    }
    if let Some(usage) = chunk.usage {
        out.push(SseChunk::Usage {
            prompt: usage.prompt_tokens,
            completion: usage.completion_tokens,
        });
    }
    Ok(out)
}

/// Open an SSE stream against llama-server and forward parsed chunks via an mpsc channel.
/// Cancellation: cancel the token → the select loop closes EventSource → llama-server
/// stops within a few tokens.
pub fn stream_chat(
    http: reqwest::Client,
    port: u16,
    body: Value,
    cancel: CancellationToken,
) -> mpsc::Receiver<Result<SseChunk, AppError>> {
    let (tx, rx) = mpsc::channel::<Result<SseChunk, AppError>>(64);

    tokio::spawn(async move {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
        let req = http
            .post(&url)
            .header("Accept", "text/event-stream")
            .json(&body);

        let mut es = match EventSource::new(req) {
            Ok(es) => es,
            Err(e) => {
                let _ = tx
                    .send(Err(AppError::ChatTransport(format!(
                        "eventsource init: {}",
                        e
                    ))))
                    .await;
                return;
            }
        };

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    es.close();
                    let _ = tx.send(Ok(SseChunk::Done)).await;
                    break;
                }
                ev = es.next() => {
                    match ev {
                        None => {
                            // Stream ended naturally
                            let _ = tx.send(Ok(SseChunk::Done)).await;
                            break;
                        }
                        Some(Err(reqwest_eventsource::Error::StreamEnded)) => {
                            let _ = tx.send(Ok(SseChunk::Done)).await;
                            break;
                        }
                        Some(Err(e)) => {
                            let _ = tx
                                .send(Err(AppError::ChatStream(format!("{}", e))))
                                .await;
                            break;
                        }
                        Some(Ok(Event::Open)) => continue,
                        Some(Ok(Event::Message(m))) => {
                            let data = m.data.trim();
                            if data == "[DONE]" {
                                let _ = tx.send(Ok(SseChunk::Done)).await;
                                break;
                            }
                            match parse_sse_data(data) {
                                Ok(chunks) => {
                                    let mut closed = false;
                                    for c in chunks {
                                        if tx.send(Ok(c)).await.is_err() {
                                            closed = true;
                                            break;
                                        }
                                    }
                                    if closed {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx
                                        .send(Err(AppError::ChatStream(format!(
                                            "parse: {} (line: {})",
                                            e, data
                                        ))))
                                        .await;
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_delta() {
        let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;
        let chunks = parse_sse_data(data).unwrap();
        assert!(matches!(&chunks[..], [SseChunk::Delta(s)] if s == "Hello"));
    }

    /// Captured shape from llama-server serving a Qwen3 thinking GGUF: the
    /// model's reasoning streams as `delta.reasoning_content` while `content`
    /// stays empty. Before the `Reasoning` variant these were dropped and a
    /// thinking-heavy turn arrived completely empty.
    #[test]
    fn parses_reasoning_delta() {
        let data = r#"{"choices":[{"delta":{"reasoning_content":"Here's a thinking process: "}}]}"#;
        let chunks = parse_sse_data(data).unwrap();
        assert!(
            matches!(&chunks[..], [SseChunk::Reasoning(s)] if s == "Here's a thinking process: ")
        );
    }

    #[test]
    fn reasoning_precedes_content_when_both_present() {
        let data = r#"{"choices":[{"delta":{"content":"answer","reasoning_content":"thought"}}]}"#;
        let chunks = parse_sse_data(data).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(matches!(&chunks[0], SseChunk::Reasoning(s) if s == "thought"));
        assert!(matches!(&chunks[1], SseChunk::Delta(s) if s == "answer"));
    }

    #[test]
    fn empty_strings_emit_nothing() {
        let data = r#"{"choices":[{"delta":{"content":"","reasoning_content":""}}]}"#;
        assert!(parse_sse_data(data).unwrap().is_empty());
    }

    #[test]
    fn finish_reason_and_usage() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":34}}"#;
        let chunks = parse_sse_data(data).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(matches!(&chunks[0], SseChunk::FinishReason(s) if s == "stop"));
        assert!(matches!(
            chunks[1],
            SseChunk::Usage { prompt: 12, completion: 34 }
        ));
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        // Real llama-server chunks carry id/object/created/model/etc.
        let data = r#"{"id":"chatcmpl-x","object":"chat.completion.chunk","created":1718000000,"model":"qwen3","choices":[{"index":0,"delta":{"reasoning_content":"hm"},"logprobs":null,"finish_reason":null}]}"#;
        let chunks = parse_sse_data(data).unwrap();
        assert!(matches!(&chunks[..], [SseChunk::Reasoning(s)] if s == "hm"));
    }
}
