//! Upgrade coverage for the destructive M17 migration. Uses only an in-memory DB.
use std::borrow::Cow;

use sqlx::sqlite::SqlitePoolOptions;

#[tokio::test]
async fn removes_agent_data_but_preserves_chat_and_fts() {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    let mut previous = sqlx::migrate!("./migrations");
    previous.migrations = Cow::Owned(
        previous
            .iter()
            .filter(|m| m.version < 14)
            .cloned()
            .collect(),
    );
    previous.run(&pool).await.unwrap();

    // Historical-schema fixtures must not be checked against today's sqlx cache.
    sqlx::raw_sql(
        r#"
        PRAGMA foreign_keys = ON;
        INSERT INTO agents (id, name, system_prompt, created_at)
            VALUES ('profile', 'Coder', 'Code', 1);
        INSERT INTO projects (id, name, path, created_at, last_used_at)
            VALUES ('project', 'Old project', 'C:/old', 1, 1);
        INSERT INTO conversations (id, title, model_id, project_id, working_dir, mode, transcript, created_at, updated_at)
            VALUES ('chat', 'Keep me', 'model', 'project', 'C:/old', 'yolo', 'chat.jsonl', 10, 20);
        INSERT INTO conversations (id, agent_id, project_id, fluid_session_id, transcript, created_at, updated_at)
            VALUES ('run', 'profile', 'project', 'session', 'run.jsonl', 1, 2);
        INSERT INTO conversations (id, fluid_session_id, transcript, created_at, updated_at)
            VALUES ('fluid-run', 'other-session', 'fluid-run.jsonl', 1, 2);
        INSERT INTO conversations (id, agent_id, parent_conversation_id, transcript, created_at, updated_at)
            VALUES ('child', 'profile', 'run', 'child.jsonl', 1, 2);
        INSERT INTO conversations (id, parent_conversation_id, transcript, created_at, updated_at)
            VALUES ('grandchild', 'child', 'grandchild.jsonl', 1, 2);
        INSERT INTO messages_index (conversation_id, seq, byte_offset, byte_len, role, ts)
            SELECT id, 0, 123, 45, 'user', 1 FROM conversations;
        INSERT INTO messages_fts (conversation_id, seq, role, text)
            SELECT id, 0, 'user', 'searchable content' FROM conversations;
        INSERT INTO conversation_settings (conversation_id, system_prompt, updated_at)
            SELECT id, 'saved prompt', 1 FROM conversations;
        INSERT INTO tool_allowlists (agent_id, project_id, tool_name, arg_pattern, decision, created_at)
            VALUES ('profile', 'project', 'Bash', '*', 'allow', 1);
        INSERT INTO mcp_servers (id, name, transport, source, created_at)
            VALUES ('server', 'Old MCP', 'stdio', 'manual', 1);
        INSERT INTO mcp_tool_consents VALUES ('server', 'tool', 'allow');
        INSERT INTO settings VALUES ('fluid_tool_style', 'legacy');
        INSERT INTO settings VALUES ('kept_setting', 'keep');
        "#,
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::migrate!("./migrations").run(&pool).await.unwrap();
    // Re-running the full migrator is a no-op, not a second deletion.
    sqlx::migrate!("./migrations").run(&pool).await.unwrap();

    let chats = super::super::chat::persistence::list_conversations(&pool)
        .await
        .unwrap();
    assert_eq!(chats.len(), 1);
    assert_eq!(chats[0].id, "chat");
    assert_eq!(chats[0].title.as_deref(), Some("Keep me"));
    assert_eq!(chats[0].model_id.as_deref(), Some("model"));
    assert_eq!(chats[0].updated_at, 20);
    assert_eq!(chats[0].message_count, 1);
    let transcript: String =
        sqlx::query_scalar("SELECT transcript FROM conversations WHERE id = 'chat'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(transcript, "chat.jsonl");
    let offsets: Vec<(String, i64, i64)> =
        sqlx::query_as("SELECT conversation_id, byte_offset, byte_len FROM messages_index")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(offsets, vec![("chat".into(), 123, 45)]);
    let hits = super::super::chat::persistence::search_messages(&pool, "searchable", 10)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].conversation_id, "chat");
    let settings: Vec<(String, String)> =
        sqlx::query_as("SELECT conversation_id, system_prompt FROM conversation_settings")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(settings, vec![("chat".into(), "saved prompt".into())]);
    let remaining_settings: Vec<(String, String)> =
        sqlx::query_as("SELECT key, value FROM settings")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(
        remaining_settings,
        vec![("kept_setting".into(), "keep".into())]
    );
    let removed_tables: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name IN
         ('agents', 'projects', 'tool_allowlists', 'mcp_servers', 'mcp_tool_consents', 'removed_agent_conversations')",
    ).fetch_one(&pool).await.unwrap();
    assert_eq!(removed_tables, 0);
    let foreign_key_errors = sqlx::query("PRAGMA foreign_key_check")
        .fetch_all(&pool)
        .await
        .unwrap();
    assert!(foreign_key_errors.is_empty());
    assert_eq!(
        sqlx::query_scalar::<_, i64>("PRAGMA foreign_keys")
            .fetch_one(&pool)
            .await
            .unwrap(),
        1
    );
}
