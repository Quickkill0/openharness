/// Split `<think>…</think>` blocks (as emitted by reasoning models like
/// DeepSeek-R1 / QwQ) out of an assistant turn. Returns `(rest, thinking)`.
/// An unclosed `<think>` (stream cut mid-reasoning) swallows the remainder.
/// Used by the chat manager as an inline-think fallback when llama-server
/// didn't split reasoning into `delta.reasoning_content`.
pub fn extract_thinking(text: &str) -> (String, Option<String>) {
    const T_OPEN: &str = "<think>";
    const T_CLOSE: &str = "</think>";
    let mut rest = text;
    let mut kept = String::new();
    let mut thinking = String::new();

    // Implicit open: chat templates that pre-open the think block (Qwen3 /
    // DeepSeek-R1 style) make the model emit `reasoning…</think>answer` with
    // no `<think>` tag at all. If the first `</think>` has no `<think>`
    // anywhere before it, everything up to it is reasoning. First-occurrence
    // rule tradeoff: a stray `</think>` in prose (e.g. quoted in a code fence)
    // *after* a legit pair is untouched, because that pair's `<think>`
    // precedes it; but a genuinely implicit turn whose answer also quotes
    // `</think>` splits at the first one — acceptable, since reasoning always
    // ends at the first close in the pre-opened template anyway.
    if let Some(close_idx) = rest.find(T_CLOSE) {
        if !rest[..close_idx].contains(T_OPEN) {
            thinking.push_str(rest[..close_idx].trim());
            thinking.push('\n');
            rest = &rest[close_idx + T_CLOSE.len()..];
        }
    }

    if !rest.contains(T_OPEN) {
        let thinking = thinking.trim().to_string();
        return (
            rest.to_string(),
            if thinking.is_empty() {
                None
            } else {
                Some(thinking)
            },
        );
    }
    while let Some(open_idx) = rest.find(T_OPEN) {
        kept.push_str(&rest[..open_idx]);
        let after = &rest[open_idx + T_OPEN.len()..];
        match after.find(T_CLOSE) {
            Some(close_idx) => {
                thinking.push_str(after[..close_idx].trim());
                thinking.push('\n');
                rest = &after[close_idx + T_CLOSE.len()..];
            }
            None => {
                thinking.push_str(after.trim());
                rest = "";
                break;
            }
        }
    }
    kept.push_str(rest);
    let thinking = thinking.trim().to_string();
    (
        kept,
        if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_thinking_splits_blocks() {
        let (rest, think) = extract_thinking("<think>reason</think>\nanswer");
        assert_eq!(rest.trim(), "answer");
        assert_eq!(think.as_deref(), Some("reason"));

        // Implicit open (pre-opened template): `reason</think>answer`.
        let (rest, think) = extract_thinking("reason here</think>the answer");
        assert_eq!(rest, "the answer");
        assert_eq!(think.as_deref(), Some("reason here"));

        // Unclosed tail swallows the remainder.
        let (rest, think) = extract_thinking("prefix <think>cut off mid");
        assert_eq!(rest, "prefix ");
        assert_eq!(think.as_deref(), Some("cut off mid"));

        // No think content at all.
        let (rest, think) = extract_thinking("plain answer");
        assert_eq!(rest, "plain answer");
        assert!(think.is_none());
    }
}
