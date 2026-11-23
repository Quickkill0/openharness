use chrono::Utc;
use serde::Serialize;
use sqlx::SqlitePool;
use uuid::Uuid;
use crate::chat::transcript::{MessageRecord, MessageMeta, TranscriptWriter};
use crate::error::AppError;

#[derive(Debug, Clone, Serialize)]
pub struct ConversationRecord {
    pub id: String,
    pub title: Option<String>,
    pub model_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConversationSummary {
    pub id: String,
    pub title: Option<String>,
    pub model_id: Option<String>,
    pub updated_at: i64,
    pub message_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub conversation_id: String,
    pub seq: i64,
    pub snippet: String,
}

pub async fn create_conversation(pool: &SqlitePool, model_id: Option<&str>) -> Result<ConversationRecord, AppError> {
    let id = Uuid::new_v4().to_string();
    let now = Utc::now().timestamp();
    let transcript = format!("{id}.jsonl");
    sqlx::query!(
        "INSERT INTO conversations (id, title, model_id, transcript, created_at, updated_at)
         VALUES (?, NULL, ?, ?, ?, ?)",
        id, model_id, transcript, now, now
    ).execute(pool).await?;
    Ok(ConversationRecord {
        id, title: None, model_id: model_id.map(|s| s.to_string()),
        created_at: now, updated_at: now,
    })
}

pub async fn append_message(
    pool: &SqlitePool,
    writer: &mut TranscriptWriter,
    conv_id: &str,
    role: &str,
    content: &str,
    ts: i64,
    tokens: Option<crate::chat::transcript::TokenCounts>,
    finish_reason: Option<String>,
) -> Result<(MessageMeta, MessageRecord), AppError> {
    // 1. JSONL append (canonical)
    let (meta, record) = writer.append(role, content, ts, tokens, finish_reason).await?;

    // 2. DB tx: messages_index + messages_fts + conversations.updated_at
    let seq_i = meta.seq as i64;
    let offset_i = meta.byte_offset as i64;
    let len_i = meta.byte_len as i64;
    let mut tx = pool.begin().await?;
    sqlx::query!(
        "INSERT INTO messages_index (conversation_id, seq, byte_offset, byte_len, role, has_tool_use, ts)
         VALUES (?, ?, ?, ?, ?, 0, ?)",
        conv_id, seq_i, offset_i, len_i, record.role, ts
    ).execute(&mut *tx).await?;

    if matches!(record.role.as_str(), "user" | "assistant") {
        sqlx::query!(
            "INSERT INTO messages_fts (conversation_id, seq, role, text) VALUES (?, ?, ?, ?)",
            conv_id, seq_i, record.role, record.content
        ).execute(&mut *tx).await?;
    }

    sqlx::query!(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        ts, conv_id
    ).execute(&mut *tx).await?;
    tx.commit().await?;

    Ok((meta, record))
}

/// Chat conversations, newest first.
pub async fn list_conversations(pool: &SqlitePool) -> Result<Vec<ConversationSummary>, AppError> {
    let rows = sqlx::query!(
        r#"SELECT c.id, c.title, c.model_id, c.updated_at,
                  COALESCE((SELECT COUNT(*) FROM messages_index mi WHERE mi.conversation_id = c.id), 0) AS message_count
           FROM conversations c
           ORDER BY c.updated_at DESC"#
    ).fetch_all(pool).await?;
    Ok(rows.into_iter().map(|r| ConversationSummary {
        id: r.id.unwrap_or_default(),
        title: r.title,
        model_id: r.model_id,
        updated_at: r.updated_at,
        message_count: r.message_count,
    }).collect())
}

pub async fn get_conversation(pool: &SqlitePool, conv_id: &str) -> Result<Option<ConversationRecord>, AppError> {
    let row = sqlx::query!(
        "SELECT id, title, model_id, created_at, updated_at FROM conversations WHERE id = ?",
        conv_id
    ).fetch_optional(pool).await?;
    Ok(row.map(|r| ConversationRecord {
        id: r.id.unwrap_or_default(),
        title: r.title,
        model_id: r.model_id,
        created_at: r.created_at,
        updated_at: r.updated_at,
    }))
}

pub async fn load_conversation_messages(pool: &SqlitePool, conv_id: &str) -> Result<Vec<MessageRecord>, AppError> {
    let row = sqlx::query!("SELECT transcript FROM conversations WHERE id = ?", conv_id)
        .fetch_optional(pool).await?
        .ok_or_else(|| AppError::ConversationNotFound(conv_id.to_string()))?;
    let path = crate::config::transcripts_dir().join(&row.transcript);
    crate::chat::transcript::read_messages(&path).await
}

pub async fn delete_conversation(pool: &SqlitePool, conv_id: &str) -> Result<(), AppError> {
    // FTS5 has no foreign key; remove its entries alongside the chat row.
    let rows = sqlx::query!(
        "SELECT id, transcript FROM conversations WHERE id = ?",
        conv_id
    ).fetch_all(pool).await?;
    let mut tx = pool.begin().await?;
    for r in &rows {
        let id = r.id.as_deref().unwrap_or_default();
        // FTS5 has no FK; manual delete
        sqlx::query!("DELETE FROM messages_fts WHERE conversation_id = ?", id).execute(&mut *tx).await?;
    }
    // CASCADE handles messages_index + conversation_settings.
    sqlx::query!("DELETE FROM conversations WHERE id = ?", conv_id).execute(&mut *tx).await?;
    tx.commit().await?;
    for r in rows {
        let path = crate::config::transcripts_dir().join(&r.transcript);
        let _ = tokio::fs::remove_file(&path).await;
    }
    Ok(())
}

/// Remove every entry with `seq >= from_seq` — rewrite the JSONL transcript
/// and delete the matching `messages_index` / `messages_fts` rows. The chat
/// actor must drop any open `TranscriptWriter` on this conversation first and
/// reopen afterwards (the writer caches `next_seq` / `byte_cursor`).
pub async fn truncate_from(pool: &SqlitePool, conv_id: &str, from_seq: u64) -> Result<(), AppError> {
    let row = sqlx::query!("SELECT transcript FROM conversations WHERE id = ?", conv_id)
        .fetch_optional(pool).await?
        .ok_or_else(|| AppError::ConversationNotFound(conv_id.to_string()))?;
    let path = crate::config::transcripts_dir().join(&row.transcript);
    truncate_from_at(pool, conv_id, &path, from_seq).await
}

/// [`truncate_from`] with an explicit transcript path (testable without the
/// app data dir).
pub async fn truncate_from_at(
    pool: &SqlitePool,
    conv_id: &str,
    path: &std::path::Path,
    from_seq: u64,
) -> Result<(), AppError> {
    let messages = crate::chat::transcript::read_messages(path).await?;
    let mut kept = String::new();
    for m in messages.iter().filter(|m| m.seq < from_seq) {
        kept.push_str(
            &serde_json::to_string(m)
                .map_err(|e| AppError::TranscriptIoError(format!("serialize: {e}")))?,
        );
        kept.push('\n');
    }
    tokio::fs::write(path, kept)
        .await
        .map_err(|e| AppError::TranscriptIoError(format!("rewrite: {e}")))?;

    let seq_i = from_seq as i64;
    let now = Utc::now().timestamp();
    let mut tx = pool.begin().await?;
    sqlx::query!(
        "DELETE FROM messages_index WHERE conversation_id = ? AND seq >= ?",
        conv_id, seq_i
    ).execute(&mut *tx).await?;
    sqlx::query!(
        "DELETE FROM messages_fts WHERE conversation_id = ? AND seq >= ?",
        conv_id, seq_i
    ).execute(&mut *tx).await?;
    sqlx::query!("UPDATE conversations SET updated_at = ? WHERE id = ?", now, conv_id)
        .execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(())
}

pub async fn derive_and_set_title(pool: &SqlitePool, conv_id: &str, first_user_content: &str) -> Result<(), AppError> {
    // Trim, take up to 60 chars at word boundary
    let trimmed = first_user_content.trim();
    let mut title = String::new();
    for ch in trimmed.chars() {
        if title.len() + ch.len_utf8() > 60 { break; }
        title.push(ch);
    }
    // Optional: trim trailing partial word
    if title.len() == 60 {
        if let Some(idx) = title.rfind(' ') {
            title.truncate(idx);
        }
    }
    let title_opt: Option<String> = if title.is_empty() { None } else { Some(title) };
    sqlx::query!(
        "UPDATE conversations SET title = ? WHERE id = ? AND title IS NULL",
        title_opt, conv_id
    ).execute(pool).await?;
    Ok(())
}

pub async fn search_messages(pool: &SqlitePool, query: &str, limit: i64) -> Result<Vec<SearchHit>, AppError> {
    // Use sqlx::query_as (untyped) because FTS5 snippet() cannot be type-checked at compile time.
    let rows = sqlx::query_as::<_, (String, i64, String)>(
        "SELECT conversation_id, seq, snippet(messages_fts, 3, '<mark>', '</mark>', '…', 16) AS s
         FROM messages_fts
         WHERE messages_fts MATCH ?1
         ORDER BY rank
         LIMIT ?2"
    )
    .bind(query)
    .bind(limit)
    .fetch_all(pool).await?;
    Ok(rows.into_iter().map(|(conversation_id, seq, snippet)| SearchHit {
        conversation_id, seq, snippet
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::transcript::TranscriptWriter;
    use sqlx::sqlite::SqlitePoolOptions;
    use tempfile::TempDir;

    async fn test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await.unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    #[tokio::test]
    async fn create_and_list() {
        let pool = test_pool().await;
        let c = create_conversation(&pool, Some("m1")).await.unwrap();
        let list = list_conversations(&pool).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, c.id);
        assert_eq!(list[0].message_count, 0);
    }

    #[tokio::test]
    async fn append_and_fts() {
        let pool = test_pool().await;
        let dir = TempDir::new().unwrap();
        let c = create_conversation(&pool, Some("m1")).await.unwrap();
        let path = dir.path().join(format!("{}.jsonl", c.id));
        let mut w = TranscriptWriter::open_or_create(path.clone()).await.unwrap();
        append_message(&pool, &mut w, &c.id, "user", "hello world", 100, None, None).await.unwrap();
        append_message(&pool, &mut w, &c.id, "assistant", "hi there", 101, None, Some("stop".into())).await.unwrap();
        let list = list_conversations(&pool).await.unwrap();
        assert_eq!(list[0].message_count, 2);
        let hits = search_messages(&pool, "hello", 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].snippet.contains("hello"));
    }

    #[tokio::test]
    async fn truncate_rewrites_transcript_and_indexes() {
        let pool = test_pool().await;
        let dir = TempDir::new().unwrap();
        let c = create_conversation(&pool, Some("m1")).await.unwrap();
        let path = dir.path().join(format!("{}.jsonl", c.id));
        let mut w = TranscriptWriter::open_or_create(path.clone()).await.unwrap();
        append_message(&pool, &mut w, &c.id, "user", "first question", 100, None, None).await.unwrap();
        append_message(&pool, &mut w, &c.id, "assistant", "first answer", 101, None, Some("stop".into())).await.unwrap();
        append_message(&pool, &mut w, &c.id, "user", "second question", 102, None, None).await.unwrap();
        append_message(&pool, &mut w, &c.id, "assistant", "second answer", 103, None, Some("stop".into())).await.unwrap();
        drop(w);

        // Cut from the second user message (seq 2) onward.
        truncate_from_at(&pool, &c.id, &path, 2).await.unwrap();

        let remaining = crate::chat::transcript::read_messages(&path).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[1].content, "first answer");

        let list = list_conversations(&pool).await.unwrap();
        assert_eq!(list[0].message_count, 2);

        // FTS rows for the removed turns are gone; kept turns still match.
        assert!(search_messages(&pool, "second", 10).await.unwrap().is_empty());
        assert_eq!(search_messages(&pool, "first", 10).await.unwrap().len(), 2);

        // A reopened writer resumes at the truncated seq.
        let mut w2 = TranscriptWriter::open_or_create(path.clone()).await.unwrap();
        let (m, _) = w2.append("user", "redo", 104, None, None).await.unwrap();
        assert_eq!(m.seq, 2);
    }

    #[tokio::test]
    async fn title_derivation() {
        let pool = test_pool().await;
        let c = create_conversation(&pool, None).await.unwrap();
        derive_and_set_title(&pool, &c.id, "Explain quantum entanglement in simple terms please thanks").await.unwrap();
        let rec = get_conversation(&pool, &c.id).await.unwrap().unwrap();
        assert!(rec.title.is_some());
        let t = rec.title.unwrap();
        assert!(t.len() <= 60);
        assert!(t.starts_with("Explain quantum"));
    }
}
