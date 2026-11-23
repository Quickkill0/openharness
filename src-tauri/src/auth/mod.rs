use crate::error::AppError;

const SERVICE: &str = "ai.openharness.app";
/// Legacy credential service from the "Hangar AI" era — read once by the
/// rebrand migration, then deleted.
const LEGACY_SERVICE: &str = "ai.hangar.app";
const USERNAME: &str = "hf_token";

/// Store the HuggingFace API token in the OS credential store
/// (Windows Credential Manager on Windows).
pub fn set_hf_token(token: &str) -> Result<(), AppError> {
    let entry = keyring::Entry::new(SERVICE, USERNAME)?;
    entry.set_password(token)?;
    Ok(())
}

/// Retrieve the stored HuggingFace API token.
/// Returns `None` if no token has been saved.
pub fn get_hf_token() -> Result<Option<String>, AppError> {
    let entry = keyring::Entry::new(SERVICE, USERNAME)?;
    match entry.get_password() {
        Ok(token) => Ok(Some(token)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(AppError::Keyring(e.to_string())),
    }
}

/// Remove the stored HuggingFace API token.
pub fn clear_hf_token() -> Result<(), AppError> {
    let entry = keyring::Entry::new(SERVICE, USERNAME)?;
    match entry.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()), // idempotent
        Err(e) => Err(AppError::Keyring(e.to_string())),
    }
}

/// One-time rebrand migration of the HF token credential.
///
/// If no token exists under the new service but one exists under the legacy
/// `ai.hangar.app` service, copy it across and delete the legacy entry.
/// Idempotent and best-effort. Returns `Ok(true)` when a token was migrated.
pub fn migrate_legacy_token() -> Result<bool, AppError> {
    if get_hf_token()?.is_some() {
        return Ok(false);
    }

    let legacy = keyring::Entry::new(LEGACY_SERVICE, USERNAME)?;
    match legacy.get_password() {
        Ok(token) => {
            set_hf_token(&token)?;
            let _ = legacy.delete_credential(); // best-effort cleanup
            Ok(true)
        }
        Err(keyring::Error::NoEntry) => Ok(false),
        Err(e) => Err(AppError::Keyring(e.to_string())),
    }
}

/// Validate the stored HF token by calling the whoami-v2 endpoint.
/// Returns the parsed whoami response body on success.
pub async fn validate_hf_token(
    client: &reqwest::Client,
) -> Result<serde_json::Value, AppError> {
    let token = get_hf_token()?.ok_or_else(|| AppError::Other("No HF token stored".into()))?;

    let resp = client
        .get("https://huggingface.co/api/whoami-v2")
        .bearer_auth(&token)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError::Other(format!(
            "HuggingFace rejected the token (HTTP {}): {}",
            status, body
        )));
    }

    let json: serde_json::Value = resp.json().await?;
    Ok(json)
}
