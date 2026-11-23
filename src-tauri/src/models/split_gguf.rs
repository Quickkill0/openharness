//! Filename identity for Store split sets. The stem includes the repo subfolder.
use crate::error::AppError;

/// Use actual destination paths, not repo/revision strings: repo-root
/// sanitization also aliases names. Match Windows separators and Unicode case
/// on every platform. Folding is conservative (it can refuse extra aliases).
pub fn destination_key(path: impl AsRef<std::path::Path>) -> String {
    path.as_ref().to_string_lossy().replace('\\', "/").split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(|part| part.trim_end_matches(['.', ' ']).to_lowercase().to_uppercase())
        .collect::<Vec<_>>().join("/")
}

pub struct Part<'a> {
    pub stem: &'a str,
    pub number: u32,
    pub count: u32,
}

pub fn part(path: &str) -> Option<Part<'_>> {
    let base = path.strip_suffix(".gguf")?;
    let (head, count) = base.rsplit_once("-of-")?;
    let (stem, number) = head.rsplit_once('-')?;
    if number.is_empty() || count.is_empty()
        || !number.bytes().all(|b| b.is_ascii_digit())
        || !count.bytes().all(|b| b.is_ascii_digit()) { return None; }
    let count = count.parse().ok()?;
    (count > 1).then_some(Part { stem, number: number.parse().ok()?, count })
}

/// Exactly one complete set, with no duplicate, auxiliary or unrelated files.
pub fn validate<'a>(paths: impl IntoIterator<Item = &'a str>) -> Result<String, AppError> {
    let paths: Vec<_> = paths.into_iter().collect();
    let invalid = || AppError::Other("Download one complete split GGUF set, including every part.".into());
    let first = paths.first().and_then(|p| part(p)).ok_or_else(invalid)?;
    let mut numbers = std::collections::HashSet::new();
    for path in &paths {
        let p = part(path).ok_or_else(invalid)?;
        if p.stem != first.stem || p.count != first.count || p.number == 0
            || p.number > p.count || !numbers.insert(p.number) { return Err(invalid()); }
    }
    if numbers.len() != first.count as usize { return Err(invalid()); }
    // llama.cpp expands siblings using the primary filename's padding.
    let primary = paths.iter().find(|p| part(p).unwrap().number == 1).ok_or_else(invalid)?;
    let (head, count) = primary.strip_suffix(".gguf").unwrap().rsplit_once("-of-").unwrap();
    let (_, no) = head.rsplit_once('-').unwrap();
    if !(1..=first.count).all(|n| paths.contains(&format!("{}-{n:0wn$}-of-{count}.gguf", first.stem, wn=no.len()).as_str())) {
        return Err(invalid());
    }
    Ok(first.stem.to_string())
}
