// Standalone fake native binary, compiled by tests with rustc (no dependencies).
use std::{io::{Read, Write}, net::TcpListener};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("--engine-info") {
        // A file beside the binary stands in for a probe that never answers.
        if std::env::current_exe().unwrap().with_file_name("probe-hangs").exists() {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
        // A file beside the binary stands in for a missing dependency.
        if std::env::current_exe().unwrap().with_file_name("probe-fails").exists() {
            eprintln!("missing dependency");
            std::process::exit(1);
        }
        // A "cuda" file beside the binary stands in for a CUDA build with a card.
        let cuda = std::env::current_exe().unwrap().with_file_name("cuda").exists();
        let (tier, backends) = if cuda { ("true", r#"[{"name":"CUDA","devices":[{"name":"CUDA0","description":"stub","type":"gpu","memory_total_mib":24564,"memory_free_mib":23000}]}]"#) } else { ("false", "[]") };
        println!(r#"{{"engine":"openharness-engine","engine_version":"stub","llama_cpp":{{"tag":"test","commit":"test"}},"features":{{"ssd_expert_streaming":true,"cuda_vram_expert_tier":{tier}}},"backends":{backends}}}"#);
        return;
    }
    let value = |key: &str| a.windows(2).find(|w| w[0] == key).map(|w| w[1].as_str()).unwrap();
    let model = value("-m");
    let card = value("-ngl") == "all";
    if !card { assert_eq!(value("-ngl"), "0"); }
    if !model.contains("legacy") {
    assert_eq!(value("-np"), "1");
    assert!(a.iter().any(|x| x == "--ssd"));
    assert!(a.iter().any(|x| x.starts_with("engine.ram_budget_mib=")));
    // Dense layers on the card come only with experts on the host.
    assert_eq!(card, a.iter().any(|x| x == "--cpu-moe"));
    let cuda = std::env::current_exe().unwrap().with_file_name("cuda").exists();
    assert!(a.iter().any(|x| if cuda { x.starts_with("vram.budget=") } else { x == "vram.budget=0" }));
    for flag in ["--fit", "--moe-cache-mib", "--n-cpu-moe", "--nextn"] { assert!(!a.iter().any(|x| x == flag)); }
    }
    if model.contains("refuse") { eprintln!("[oh-ssd] unsupported expert layout: F32 experts are refused"); std::process::exit(1); }
    assert!(model.contains("-00001-of-"), "must start from first split: {model}");
    // Small real anonymous cache so the production resident-private probe is
    // exercised. Never allocate a model or depend on available machine RAM.
    let mut cache = vec![0u8; if model.contains("warm") { 64 * 1024 * 1024 } else { 0 }];
    for byte in cache.iter_mut().step_by(4096) { *byte = 1; }
    let listener = TcpListener::bind(format!("127.0.0.1:{}", value("--port"))).unwrap();
    let mut health = 0;
    for socket in listener.incoming() {
        std::hint::black_box(&cache);
        let mut socket = socket.unwrap();
        let mut request = Vec::new();
        let mut buf = [0; 4096];
        socket.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
        loop {
            let n = socket.read(&mut buf).unwrap();
            if n == 0 { break; }
            request.extend_from_slice(&buf[..n]);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&request[..end]);
                let len: usize = header.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|n| n.trim().parse().unwrap())).unwrap_or(0);
                if request.len() >= end + 4 + len { break; }
            }
        }
        let req = String::from_utf8_lossy(&request);
        let mut status = "200 OK";
        let mut kind = "application/json";
        if req.starts_with("GET /test/exit ") { std::process::exit(1); }
        let body = if req.starts_with("GET /health ") {
            health += 1;
            if health == 1 || (model.contains("slow") && !std::path::Path::new(&format!("{model}.ready")).exists()) { status = "503 Service Unavailable"; r#"{"status":"loading"}"# } else { r#"{"status":"ok"}"# }
        } else if req.starts_with("GET /v1/models ") {
            r#"{"object":"list","data":[{"id":"glm-native","object":"model","owned_by":"local"}]}"#
        } else if req.starts_with("POST /v1/embeddings ") {
            status = "400 Bad Request"; r#"{"error":{"message":"Embeddings are not supported by this generation model","type":"invalid_request_error"}}"#
        } else if req.starts_with("GET /metrics ") {
            kind = "text/plain"; "llamacpp:tokens_predicted_total 500\nllamacpp:tokens_predicted_seconds_total 222.2\n"
        } else if req.starts_with("POST /v1/chat/completions ") {
            if req.contains("\"stream\":true") || req.contains("\"stream\": true") {
                kind = "text/event-stream";
                "data: {\"choices\":[{\"delta\":{\"content\":\"native stub reply\"}}]}\n\ndata: [DONE]\n\n"
            } else { r#"{"choices":[{"message":{"role":"assistant","content":"native stub reply"}}],"usage":{"prompt_tokens":1,"completion_tokens":3}}"# }
        } else { status = "404 Not Found"; r#"{"error":{"message":"Unsupported route"}}"# };
        write!(socket, "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        eprintln!("stub served {}", req.lines().next().unwrap_or_default());
    }
}
