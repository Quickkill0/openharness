//! Rust side of the JSON-Lines stdio protocol.
//!
//! Mirrors `worker-py/openharness_worker/protocol.py`. Requests carry an
//! `id` for correlation; incoming lines with an `id` are responses, lines
//! with an `event` field are unsolicited events.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkerRequest {
    Ping,
    Capabilities,
    Submit {
        task: String,
        /// Generated Rust-side so the job row can be persisted before submit.
        job_id: String,
        params: Value,
    },
    /// Load a pipeline's model into memory without running a job (W1 §16).
    /// Accepted like a submit; completion arrives as a `job_done` event whose
    /// meta reports NVML-actual VRAM where the pipeline can measure it.
    Preload {
        task: String,
        job_id: String,
        params: Value,
    },
    Cancel {
        job_id: String,
    },
    Unload {
        task: Option<String>,
    },
    Shutdown,
}

/// A request envelope as it goes over the wire: the body plus the
/// correlation id.
#[derive(Debug, Serialize)]
pub struct RequestEnvelope<'a> {
    pub id: &'a str,
    #[serde(flatten)]
    pub body: &'a WorkerRequest,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerResponse {
    pub id: String,
    pub ok: bool,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum WorkerEvent {
    JobProgress {
        job_id: String,
        pct: f32,
        stage: String,
        #[serde(default)]
        step: Option<u32>,
        #[serde(default)]
        total_steps: Option<u32>,
    },
    JobDone {
        job_id: String,
        outputs: Vec<String>,
        #[serde(default)]
        meta: Value,
        /// Which pipeline produced the job — lets the orchestrator attribute
        /// VRAM to the right resident without tracking job_id→task itself.
        #[serde(default)]
        task: Option<String>,
    },
    JobError {
        job_id: String,
        error: String,
    },
    Log {
        level: String,
        msg: String,
    },
}

/// One parsed line from the worker's stdout.
#[derive(Debug)]
pub enum Incoming {
    Response(WorkerResponse),
    Event(WorkerEvent),
}

/// Parse a stdout line. Returns `None` for lines that are neither a valid
/// response nor a valid event (the caller logs and moves on — a bad line
/// must never kill the reader).
pub fn parse_line(line: &str) -> Option<Incoming> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    if v.get("id").is_some() {
        serde_json::from_value::<WorkerResponse>(v).ok().map(Incoming::Response)
    } else if v.get("event").is_some() {
        serde_json::from_value::<WorkerEvent>(v).ok().map(Incoming::Event)
    } else {
        None
    }
}

/// Serialize a request envelope to its wire line (no trailing newline).
pub fn encode_request(id: &str, body: &WorkerRequest) -> Result<String, serde_json::Error> {
    serde_json::to_string(&RequestEnvelope { id, body })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_ping_has_id_and_op() {
        let line = encode_request("abc", &WorkerRequest::Ping).unwrap();
        let v: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["id"], "abc");
        assert_eq!(v["op"], "ping");
    }

    #[test]
    fn encode_submit_flattens_fields() {
        let req = WorkerRequest::Submit {
            task: "echo".into(),
            job_id: "j1".into(),
            params: serde_json::json!({"steps": 3}),
        };
        let v: Value = serde_json::from_str(&encode_request("r1", &req).unwrap()).unwrap();
        assert_eq!(v["op"], "submit");
        assert_eq!(v["task"], "echo");
        assert_eq!(v["job_id"], "j1");
        assert_eq!(v["params"]["steps"], 3);
    }

    #[test]
    fn encode_preload_flattens_fields() {
        let req = WorkerRequest::Preload {
            task: "text_to_image".into(),
            job_id: "p1".into(),
            params: serde_json::json!({"model": "C:\\m\\sdxl"}),
        };
        let v: Value = serde_json::from_str(&encode_request("r9", &req).unwrap()).unwrap();
        assert_eq!(v["op"], "preload");
        assert_eq!(v["task"], "text_to_image");
        assert_eq!(v["job_id"], "p1");
        assert_eq!(v["params"]["model"], "C:\\m\\sdxl");
    }

    #[test]
    fn parse_response_ok() {
        let line = r#"{"id":"r1","ok":true,"result":{"pong":true}}"#;
        match parse_line(line) {
            Some(Incoming::Response(r)) => {
                assert!(r.ok);
                assert_eq!(r.result.unwrap()["pong"], true);
            }
            other => panic!("expected response, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_err() {
        let line = r#"{"id":"r2","ok":false,"error":"boom"}"#;
        match parse_line(line) {
            Some(Incoming::Response(r)) => {
                assert!(!r.ok);
                assert_eq!(r.error.as_deref(), Some("boom"));
            }
            other => panic!("expected response, got {other:?}"),
        }
    }

    #[test]
    fn parse_progress_event() {
        let line = r#"{"event":"job_progress","job_id":"j1","pct":40.0,"stage":"echoing","step":2,"total_steps":5}"#;
        match parse_line(line) {
            Some(Incoming::Event(WorkerEvent::JobProgress { job_id, pct, step, .. })) => {
                assert_eq!(job_id, "j1");
                assert_eq!(pct, 40.0);
                assert_eq!(step, Some(2));
            }
            other => panic!("expected progress event, got {other:?}"),
        }
    }

    #[test]
    fn parse_done_event_with_null_meta_fields() {
        let line = r#"{"event":"job_done","job_id":"j1","outputs":["C:\\x\\a.png"],"meta":{}}"#;
        match parse_line(line) {
            Some(Incoming::Event(WorkerEvent::JobDone { outputs, .. })) => {
                assert_eq!(outputs.len(), 1);
            }
            other => panic!("expected done event, got {other:?}"),
        }
    }

    #[test]
    fn parse_garbage_returns_none() {
        assert!(parse_line("not json").is_none());
        assert!(parse_line(r#"{"neither":"nor"}"#).is_none());
    }
}
