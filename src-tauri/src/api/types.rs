use std::collections::BTreeMap;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BindAddress {
    #[default]
    ThisPc,
    HomeNetwork,
}
impl BindAddress {
    pub fn ip(self) -> &'static str {
        match self {
            Self::ThisPc => "127.0.0.1",
            Self::HomeNetwork => "0.0.0.0",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerSettings {
    pub enabled: bool,
    pub port: u16,
    pub bind: BindAddress,
    pub require_key: bool,
    pub auto_load: bool,
    pub idle_minutes: u32,
    pub aliases: BTreeMap<String, String>,
}
impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            port: 1234,
            bind: BindAddress::ThisPc,
            require_key: false,
            auto_load: false,
            idle_minutes: 0,
            aliases: BTreeMap::new(),
        }
    }
}
impl ServerSettings {
    pub fn auth_required(&self) -> bool {
        self.require_key || self.bind == BindAddress::HomeNetwork
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ApiModel {
    pub id: String,
    pub name: String,
    pub alias: String,
    pub loaded: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ServerStatus {
    pub running: bool,
    pub settings: ServerSettings,
    pub key_configured: bool,
    pub error: Option<String>,
    pub models: Vec<ApiModel>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConnectionInfo {
    pub this_pc: String,
    pub lan: Vec<String>,
    pub auth_required: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RequestLog {
    pub time: String,
    pub client_ip: String,
    pub model: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub tokens_per_second: Option<f64>,
    pub status: u16,
    pub duration_ms: u64,
}

#[derive(Clone, Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub code: &'static str,
}
impl ApiError {
    pub fn new(status: StatusCode, message: impl Into<String>, code: &'static str) -> Self {
        Self {
            status,
            message: message.into(),
            code,
        }
    }
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            message,
            "server_unavailable",
        )
    }
    pub fn missing() -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "Model not loaded. Load it in Models or enable auto-load on request.",
            "model_not_found",
        )
    }
    pub fn value(&self) -> serde_json::Value {
        json!({"error": {"message": self.message, "type": if self.status.is_server_error() { "server_error" } else { "invalid_request_error" }, "param": null, "code": self.code}})
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.value())).into_response()
    }
}
impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

pub fn slug(name: &str) -> String {
    let mut result = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            result.push(c.to_ascii_lowercase());
        } else if !result.is_empty() && !result.ends_with('-') {
            result.push('-');
        }
    }
    let result = result.trim_end_matches('-');
    if result.is_empty() {
        "model".into()
    } else {
        result.into()
    }
}
