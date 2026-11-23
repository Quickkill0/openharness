use std::time::Duration;

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::AppError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HfModelCard {
    pub id: String,
    pub author: Option<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub downloads: u64,
    pub likes: u64,
    pub trending_score: Option<f64>,
    pub gated: bool,
    pub private: bool,
    pub last_modified: Option<String>,
    pub created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HfFile {
    pub path: String,
    /// Resolved size: uses `lfs.size` when present, otherwise top-level `size`.
    pub size: u64,
    /// SHA-256 digest from the LFS pointer (only present for LFS-tracked files).
    pub sha256: Option<String>,
    pub is_lfs: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HfGgufMeta {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub total_params: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HfModelDetails {
    pub card: HfModelCard,
    /// Raw README markdown, truncated to 50 KB. `None` if no README exists.
    pub readme: Option<String>,
    /// GGUF metadata when the repository is a GGUF model repo.
    pub gguf: Option<HfGgufMeta>,
    /// Resolved file list.
    pub siblings: Vec<HfFile>,
    /// Repo layout classification + the file subset a repo download fetches.
    pub layout: crate::models::layout::RepoLayout,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HfSearchParams {
    pub query: Option<String>,
    /// One of: text-generation | embedding | vision-text | speech | image | any
    pub capability: Option<String>,
    /// Raw HF `pipeline_tag` passthrough (e.g. "text-to-video"). When set,
    /// supersedes `capability` — the M9 store tabs filter with this.
    #[serde(default)]
    pub task: Option<String>,
    /// One of: gguf | transformers | diffusers | safetensors | any
    pub library: Option<String>,
    /// One of: trending | downloads | likes | recent  (default: trending)
    pub sort: Option<String>,
    /// 1-based page number.
    pub page: u32,
    /// Items per page (default 30, max 50).
    pub per_page: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HfSearchResponse {
    pub results: Vec<HfModelCard>,
    pub page: u32,
    pub has_next: bool,
}

// ---------------------------------------------------------------------------
// Internal response shapes (serde only — never exposed to JS)
// ---------------------------------------------------------------------------

/// Partial shape of the HF `/api/models` list entry.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawModelEntry {
    #[serde(rename = "id")]
    id: Option<String>,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    author: Option<String>,
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    trending_score: Option<f64>,
    #[serde(default)]
    gated: Value, // can be bool or string "auto" / "manual"
    #[serde(default)]
    private: bool,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: Option<String>,
}

impl RawModelEntry {
    fn resolved_id(&self) -> String {
        self.id
            .clone()
            .or_else(|| self.model_id.clone())
            .unwrap_or_default()
    }

    fn is_gated(&self) -> bool {
        match &self.gated {
            Value::Bool(b) => *b,
            Value::String(s) => !s.is_empty() && s != "false",
            _ => false,
        }
    }
}

/// Partial shape for the `/api/models/{id}` single-model response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawModelDetail {
    #[serde(rename = "id")]
    id: Option<String>,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    author: Option<String>,
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    trending_score: Option<f64>,
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    private: bool,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: Option<String>,
    #[serde(default)]
    siblings: Vec<RawSibling>,
    gguf: Option<Value>,
}

impl RawModelDetail {
    fn resolved_id(&self) -> String {
        self.id
            .clone()
            .or_else(|| self.model_id.clone())
            .unwrap_or_default()
    }

    fn is_gated(&self) -> bool {
        match &self.gated {
            Value::Bool(b) => *b,
            Value::String(s) => !s.is_empty() && s != "false",
            _ => false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawSibling {
    #[serde(rename = "rfilename")]
    rfilename: Option<String>,
    size: Option<u64>,
    lfs: Option<RawLfs>,
}

#[derive(Debug, Deserialize)]
struct RawLfs {
    size: Option<u64>,
    #[serde(rename = "sha256")]
    sha256: Option<String>,
    #[serde(rename = "oid")]
    oid: Option<String>,
}

/// Shape returned by the tree API.
#[derive(Debug, Deserialize)]
struct RawTreeEntry {
    path: Option<String>,
    size: Option<u64>,
    lfs: Option<RawLfs>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const USER_AGENT: &str = "OpenHarness/0.1 (+https://github.com/openharness/openharness)";

/// Send a fully-built `RequestBuilder`, retrying once on HTTP 429.
///
/// The builder is cloned with `try_clone()` before the first send so the
/// second attempt has an identical request. If `try_clone()` returns `None`
/// (i.e. the request body is a stream that cannot be replayed) the function
/// returns the first 429 response as `AppError::RateLimited` rather than
/// panicking.
///
/// On a second consecutive 429 the function also returns `AppError::RateLimited`.
/// Any other non-success status is converted by `map_status`.
async fn hf_request_with_retry(req: reqwest::RequestBuilder) -> Result<reqwest::Response, AppError> {
    // Clone before consuming so we can replay on 429.
    let retry_req = req.try_clone();

    let resp = req.send().await?;

    if resp.status() != StatusCode::TOO_MANY_REQUESTS {
        return map_status(resp).await;
    }

    // 429 — read Retry-After and wait.
    let wait_secs = resp
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5);
    tokio::time::sleep(Duration::from_secs(wait_secs.min(30))).await;

    let Some(retry) = retry_req else {
        return Err(AppError::RateLimited);
    };
    let retry_resp = retry.send().await?;
    if retry_resp.status() == StatusCode::TOO_MANY_REQUESTS {
        return Err(AppError::RateLimited);
    }
    map_status(retry_resp).await
}

/// Perform a GET, handling 429 (retry-once), 401, 403, and other errors.
async fn hf_get(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
) -> Result<reqwest::Response, AppError> {
    let mut req = client.get(url).header("User-Agent", USER_AGENT);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    hf_request_with_retry(req).await
}

async fn map_status(resp: reqwest::Response) -> Result<reqwest::Response, AppError> {
    let status = resp.status();
    match status {
        s if s.is_success() => Ok(resp),
        StatusCode::UNAUTHORIZED => Err(AppError::HfAuthRequired),
        StatusCode::FORBIDDEN => Err(AppError::HfAccessDenied),
        other => {
            let code = other.as_u16();
            let body = resp.text().await.unwrap_or_default();
            let snippet: String = body.chars().take(200).collect();
            Err(AppError::HttpStatus(code, snippet))
        }
    }
}

fn map_capability_to_pipeline(capability: &str) -> (Option<&'static str>, Option<&'static str>) {
    // Returns (pipeline_tag value, extra filter value)
    match capability {
        "text-generation" => (Some("text-generation"), None),
        "embedding" => (Some("feature-extraction"), None),
        "vision-text" => (None, Some("image-text-to-text")),
        "speech" => (Some("automatic-speech-recognition"), None),
        "image" => (Some("text-to-image"), None),
        _ => (None, None), // "any"
    }
}

fn map_sort(sort: &str) -> &'static str {
    match sort {
        "downloads" => "downloads",
        "likes" => "likes",
        "recent" => "lastModified",
        _ => "trendingScore", // default: trending
    }
}

fn parse_gguf_meta(gguf_val: Option<&Value>) -> Option<HfGgufMeta> {
    let v = gguf_val?;
    Some(HfGgufMeta {
        architecture: v["architecture"].as_str().map(|s| s.to_owned()),
        context_length: v["context_length"].as_u64(),
        total_params: v["total_params"].as_u64(),
    })
}

fn sibling_to_hf_file(s: &RawSibling) -> HfFile {
    let path = s.rfilename.clone().unwrap_or_default();
    if let Some(lfs) = &s.lfs {
        let size = lfs.size.unwrap_or(0);
        let sha256 = lfs.sha256.clone().or_else(|| lfs.oid.clone());
        HfFile { path, size, sha256, is_lfs: true }
    } else {
        HfFile { path, size: s.size.unwrap_or(0), sha256: None, is_lfs: false }
    }
}

fn raw_entry_to_card(e: RawModelEntry) -> HfModelCard {
    let gated = e.is_gated();
    let id = e.resolved_id();
    HfModelCard {
        id,
        author: e.author,
        pipeline_tag: e.pipeline_tag,
        library_name: e.library_name,
        tags: e.tags,
        downloads: e.downloads,
        likes: e.likes,
        trending_score: e.trending_score,
        gated,
        private: e.private,
        last_modified: e.last_modified,
        created_at: e.created_at,
    }
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/// Search HuggingFace model catalog.
pub async fn search_models(
    client: &reqwest::Client,
    params: HfSearchParams,
    token: Option<&str>,
) -> Result<HfSearchResponse, AppError> {
    let per_page = params.per_page.min(50).max(1);
    // Fetch one extra to detect has_next without a separate count request.
    let fetch_count = per_page + 1;
    let skip = params.page.saturating_sub(1) * per_page;

    let sort_key = map_sort(params.sort.as_deref().unwrap_or("trending"));
    // Raw task passthrough (M9) supersedes the legacy capability mapping.
    let (pipeline_tag, extra_filter) = if let Some(task) = params.task.as_deref() {
        if task == "any" {
            (None, None)
        } else {
            (Some(task), None)
        }
    } else {
        let capability = params.capability.as_deref().unwrap_or("any");
        map_capability_to_pipeline(capability)
    };

    // Build query string manually so we control which params are included.
    // reqwest's RequestBuilder::query() is variadic but requires owned pairs;
    // building as a Vec avoids the urlencoding crate dependency.
    let mut qp: Vec<(&str, String)> = vec![
        ("sort", sort_key.to_string()),
        ("direction", "-1".to_string()),
        ("limit", fetch_count.to_string()),
        ("skip", skip.to_string()),
        ("full", "true".to_string()),
        ("config", "true".to_string()),
    ];

    if let Some(q) = &params.query {
        let trimmed = q.trim();
        if !trimmed.is_empty() {
            qp.push(("search", trimmed.to_string()));
        }
    }

    if let Some(tag) = pipeline_tag {
        qp.push(("pipeline_tag", tag.to_string()));
    }

    if let Some(filter) = extra_filter {
        qp.push(("filter", filter.to_string()));
    }

    if let Some(lib) = &params.library {
        if lib != "any" {
            // HF's `library=` query parameter only works for canonical libraries
            // (transformers, diffusers, sentence-transformers, …). Format names
            // like gguf/safetensors are tags — use `filter=` for those.
            if lib == "gguf" || lib == "safetensors" {
                qp.push(("filter", lib.clone()));
            } else {
                qp.push(("library", lib.clone()));
            }
        }
    }

    // reqwest handles percent-encoding for query params automatically.
    let mut req = client
        .get("https://huggingface.co/api/models")
        .header("User-Agent", USER_AGENT);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    req = req.query(&qp);

    let ok_resp = hf_request_with_retry(req).await?;
    finish_search(ok_resp, params.page, per_page).await
}

async fn finish_search(
    resp: reqwest::Response,
    page: u32,
    per_page: u32,
) -> Result<HfSearchResponse, AppError> {
    let entries: Vec<RawModelEntry> = resp.json().await?;
    let has_next = entries.len() > per_page as usize;
    let results: Vec<HfModelCard> = entries
        .into_iter()
        .take(per_page as usize)
        .map(raw_entry_to_card)
        .collect();

    Ok(HfSearchResponse { results, page, has_next })
}

/// Fetch full details for a single model, including README and file siblings.
pub async fn get_model_details(
    client: &reqwest::Client,
    repo_id: &str,
    token: Option<&str>,
) -> Result<HfModelDetails, AppError> {
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let resp = hf_get(client, &url, token).await?;
    let detail: RawModelDetail = resp.json().await?;

    let gguf = parse_gguf_meta(detail.gguf.as_ref());
    let siblings: Vec<HfFile> = detail.siblings.iter().map(sibling_to_hf_file).collect();
    let gated = detail.is_gated();

    let card = HfModelCard {
        id: detail.resolved_id(),
        author: detail.author,
        pipeline_tag: detail.pipeline_tag,
        library_name: detail.library_name,
        tags: detail.tags,
        downloads: detail.downloads,
        likes: detail.likes,
        trending_score: detail.trending_score,
        gated,
        private: detail.private,
        last_modified: detail.last_modified,
        created_at: detail.created_at,
    };

    let readme = fetch_readme(client, repo_id, token).await?;
    let layout = crate::models::layout::classify(&siblings);

    Ok(HfModelDetails { card, readme, gguf, siblings, layout })
}

/// Fetch the flat file tree for `repo_id` at `revision`.
pub async fn get_model_tree(
    client: &reqwest::Client,
    repo_id: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<Vec<HfFile>, AppError> {
    let url = format!(
        "https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=false&expand=false"
    );
    let resp = hf_get(client, &url, token).await?;
    let entries: Vec<RawTreeEntry> = resp.json().await?;

    let files = entries
        .into_iter()
        .map(|e| {
            let path = e.path.unwrap_or_default();
            if let Some(lfs) = e.lfs {
                let size = lfs.size.unwrap_or(0);
                let sha256 = lfs.sha256.or(lfs.oid);
                HfFile { path, size, sha256, is_lfs: true }
            } else {
                HfFile { path, size: e.size.unwrap_or(0), sha256: None, is_lfs: false }
            }
        })
        .collect();

    Ok(files)
}

// ---------------------------------------------------------------------------
// README helper
// ---------------------------------------------------------------------------

const README_MAX_BYTES: usize = 50 * 1024; // 50 KB

/// Slice `text` to at most `max_bytes` bytes, stepping back to a valid UTF-8
/// char boundary so no codepoint is split.
fn truncate_utf8_safe(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

async fn fetch_readme(
    client: &reqwest::Client,
    repo_id: &str,
    token: Option<&str>,
) -> Result<Option<String>, AppError> {
    let url = format!("https://huggingface.co/{repo_id}/raw/main/README.md");
    let mut req = client.get(&url).header("User-Agent", USER_AGENT);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await?;

    match resp.status() {
        StatusCode::NOT_FOUND => return Ok(None),
        s if s.is_success() => {}
        other => {
            tracing::warn!("README fetch returned {other} for {repo_id}");
            return Ok(None);
        }
    }

    // Decode lossy first (full body) so we operate on valid UTF-8, then
    // truncate at a char boundary to avoid splitting a codepoint.
    let bytes = resp.bytes().await?;
    let full = String::from_utf8_lossy(&bytes).into_owned();
    let text = truncate_utf8_safe(&full, README_MAX_BYTES).to_string();
    Ok(Some(text))
}
