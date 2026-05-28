#![allow(dead_code)]
use std::future::Future;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use time::UtcDateTime;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("limit must be non-zero")]
    InvalidLimit,

    #[error("invalid snowflake '{0}': {1}")]
    InvalidSnowflake(String, #[source] std::num::ParseIntError),

    #[error("invalid timestamp: {0}")]
    InvalidTimestamp(String),

    #[error("timestamp predates the Discord epoch")]
    TimestampTooEarly,

    #[error("worker_id must be ≤ 1023")]
    InvalidWorkerId,

    #[error("sequence must be ≤ 4095")]
    InvalidSequence,

    #[error("timestamp too large for a Discord snowflake")]
    TimestampTooLarge,

    #[error("expected yyyyMM, got {0}")]
    InvalidMonthFormat(String),

    #[error("month {0} is out of range 1–12")]
    MonthOutOfRange(u8),

    #[error("DISCORD_TOKEN not set")]
    MissingToken,

    #[error("DISCORD_CHANNEL_IDS not set")]
    MissingChannels,

    #[error("LOOKBACK_MINUTES must be an integer: {0}")]
    InvalidLookback(#[source] std::num::ParseIntError),

    #[error("channel {0} has no guild_id")]
    MissingGuildId(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[error(transparent)]
    TimeComponentRange(#[from] time::error::ComponentRange),

    #[error(transparent)]
    TimeFormat(#[from] time::error::Format),

    #[error("URL exclusion filter error: {0}")]
    AhoCorasick(#[from] aho_corasick::BuildError),

    #[error("other error: {0}")]
    Other(String),

    #[error("rate limited, retry after {retry_after:?}")]
    RateLimited { retry_after: Duration },

    #[error("HTTP status {0}")]
    HttpStatus(u16),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Reqwest error: {0}")]
    Reqwest(String),
}

const DISCORD_API: &str = "https://discord.com/api/v10";

// ---------------------------------------------------------------------------
// HttpClient trait
//
// Implementors receive a fully-formed URL and must return a deserialized T.
// Auth headers, base URLs, retries, etc. are all the implementor's concern —
// DiscordClient only hands over a complete URL and expects a T back.
//
// Bounds:
//   Clone  — DiscordClient needs to be Clone (spawned across tasks).
//   Send + Sync + 'static — required for use with tokio::spawn.
// ---------------------------------------------------------------------------

pub trait HttpClient: Clone + Send + Sync + 'static {
    /// Perform a GET request to `url` and deserialize the response body as `T`.
    fn get_json<T>(&self, url: &str) -> impl Future<Output = Result<T>> + Send
    where
        T: serde::de::DeserializeOwned + Send;
}

#[derive(Clone)]
pub struct DiscordClient<H: HttpClient> {
    http: H,
    base_url: String,
}

impl<H: HttpClient> DiscordClient<H> {
    /// Construct from any `HttpClient` implementor.
    /// The caller is responsible for baking auth into `http` beforehand.
    pub fn new(http: H) -> Self {
        Self {
            http,
            base_url: DISCORD_API.to_string(),
        }
    }

    /// Convenience: override the base URL (useful for testing against a mock server).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    async fn get_json<T>(&self, endpoint: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned + Send,
    {
        self.http
            .get_json(&format!("{}{}", self.base_url, endpoint))
            .await
    }

    pub async fn get_channel(&self, channel_id: &str) -> Result<Channel> {
        self.get_json(&format!("/channels/{channel_id}")).await
    }

    pub async fn get_guild(&self, guild_id: &str) -> Result<Guild> {
        self.get_json(&format!("/guilds/{guild_id}")).await
    }

    pub async fn get_messages(&self, channel_id: &str, limit: u8) -> Result<Vec<Message>> {
        if limit == 0 {
            return Err(Error::InvalidLimit);
        }
        self.get_json(&format!("/channels/{channel_id}/messages?limit={limit}"))
            .await
    }

    pub async fn get_messages_before(
        &self,
        channel_id: &str,
        before_id: &str,
        limit: u8,
    ) -> Result<Vec<Message>> {
        if limit == 0 {
            return Err(Error::InvalidLimit);
        }
        self.get_json(&format!(
            "/channels/{channel_id}/messages?before={before_id}&limit={limit}"
        ))
        .await
    }

    /// Fetch all messages in `date_range`, up to `limit` total.
    ///
    /// Discord returns messages newest-first.  `take_while` is therefore
    /// correct: we walk forward, stopping as soon as a message falls outside
    /// the window (all subsequent messages are older and also outside it).
    pub async fn get_messages_range(
        &self,
        channel_id: &str,
        date_range: impl std::ops::RangeBounds<UtcDateTime> + Clone,
        limit: Option<usize>,
    ) -> Result<Vec<Message>> {
        let filter_messages = |msgs: Vec<Message>| -> Vec<Message> {
            msgs.into_iter()
                .take_while(|m| {
                    m.timestamp()
                        .map(|t| date_range.contains(&t))
                        .unwrap_or(false)
                })
                .collect()
        };

        let batch_size = |collected: usize| -> u8 {
            limit
                .map(|l| l.saturating_sub(collected).min(100))
                .unwrap_or(100) as u8
        };

        // --- Initial batch ---------------------------------------------------
        let mut messages = filter_messages(match date_range.end_bound() {
            std::ops::Bound::Included(&d) | std::ops::Bound::Excluded(&d) => {
                let before_id = utils::unix_ms_to_snowflake(d.unix_timestamp() * 1000, 0, 0)?;
                self.get_messages_before(channel_id, &before_id, batch_size(0))
                    .await?
            }
            std::ops::Bound::Unbounded => self.get_messages(channel_id, batch_size(0)).await?,
        });

        if messages.is_empty() || messages.len() < 100 {
            return Ok(messages);
        }

        tracing::info!("More than 100 messages; fetching further batches");

        let deadline = Instant::now() + Duration::from_secs(60 * 5);

        loop {
            if limit.is_some_and(|l| messages.len() >= l) {
                break;
            }
            if Instant::now() >= deadline {
                tracing::warn!("get_messages_range hit 5-minute safety timeout");
                break;
            }

            // Oldest message in the current set; both checks below need it.
            let last = messages.last().unwrap(); // non-empty: checked above
            if !date_range.contains(&last.timestamp()?) {
                break;
            }
            let last_id = last.id.clone();

            let batch = filter_messages(
                self.get_messages_before(channel_id, &last_id, batch_size(messages.len()))
                    .await?,
            );
            if batch.is_empty() {
                break;
            }
            messages.extend(batch);
        }

        Ok(messages)
    }
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct Channel {
    pub id: String,
    pub name: String,
    pub guild_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Guild {
    pub id: String,
    pub name: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct User {
    pub id: String,
    pub username: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Message {
    pub id: String,
    pub content: String,
    pub author: User,
}

impl Message {
    pub fn timestamp(&self) -> Result<UtcDateTime> {
        utils::snowflake_to_utc_datetime(&self.id)
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

pub mod utils {
    use crate::{Error, Result};
    use time::{Date, Month, UtcDateTime};

    const DISCORD_EPOCH: i64 = 1_420_070_400_000;

    pub fn snowflake_to_unix_ms(s: &str) -> Result<i64> {
        let snowflake: u64 = s
            .parse()
            .map_err(|e| Error::InvalidSnowflake(s.to_string(), e))?;
        Ok((snowflake >> 22) as i64 + DISCORD_EPOCH)
    }

    pub fn snowflake_to_utc_datetime(s: &str) -> Result<UtcDateTime> {
        let ms = snowflake_to_unix_ms(s)?;
        UtcDateTime::from_unix_timestamp(ms / 1000)
            .map_err(|e| Error::InvalidTimestamp(e.to_string()))
    }

    pub fn unix_ms_to_snowflake(
        timestamp_ms: i64,
        worker_id: u16,
        sequence: u16,
    ) -> Result<String> {
        if timestamp_ms < DISCORD_EPOCH {
            return Err(Error::TimestampTooEarly);
        }
        if worker_id > 0x3FF {
            return Err(Error::InvalidWorkerId);
        }
        if sequence > 0xFFF {
            return Err(Error::InvalidSequence);
        }
        let offset = (timestamp_ms - DISCORD_EPOCH) as u64;
        if offset >> 42 != 0 {
            return Err(Error::TimestampTooLarge);
        }
        let snowflake = (offset << 22) | ((worker_id as u64) << 12) | (sequence as u64);
        Ok(snowflake.to_string())
    }

    pub fn parse_month(s: &str) -> Result<Date> {
        let s = s.replace('-', "");
        if s.len() != 6 {
            return Err(Error::InvalidMonthFormat(s));
        }
        let year: i32 = s[0..4].parse()?;
        let month: u8 = s[4..6].parse()?;
        if !(1..=12).contains(&month) {
            return Err(Error::MonthOutOfRange(month));
        }
        Ok(Date::from_calendar_date(year, Month::try_from(month)?, 1)?)
    }
}

// ---------------------------------------------------------------------------
// Config + entry point
// ---------------------------------------------------------------------------

pub struct Config {
    pub discord_token: String,
    pub channel_ids: String,
    pub lookback_minutes: i64,
    pub output_path: Option<std::path::PathBuf>,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            discord_token: std::env::var("DISCORD_TOKEN").map_err(|_| Error::MissingToken)?,
            channel_ids: std::env::var("DISCORD_CHANNEL_IDS")
                .map_err(|_| Error::MissingChannels)?,
            lookback_minutes: std::env::var("LOOKBACK_MINUTES")
                .unwrap_or_else(|_| "60".into())
                .parse()
                .map_err(|e| Error::InvalidLookback(e))?,
            output_path: std::env::var("OUTPUT_PATH").ok().map(Into::into),
        })
    }
}

// ---------------------------------------------------------------------------
// run() — still concrete (uses DefaultDiscordClient).
// Pull it into a generic fn if you need to inject a custom client in tests.
// ---------------------------------------------------------------------------

const EXCLUDED_PATTERNS: &[&str] = &[
    "cdn.",
    "tenor.",
    "redgifs.",
    "discordapp.",
    "redd.it",
    "media.tumblr.",
];

fn build_excluder() -> Result<aho_corasick::AhoCorasick> {
    aho_corasick::AhoCorasick::builder()
        .ascii_case_insensitive(true)
        .build(EXCLUDED_PATTERNS)
        .map_err(Error::from)
}

use futures::{StreamExt, stream};

/// Separate so tests can inject a mock `HttpClient`.
pub async fn run_with_client<H: HttpClient>(cfg: &Config, client: DiscordClient<H>) -> Result<()> {
    let now = UtcDateTime::now();
    let since = now.saturating_sub(time::Duration::minutes(cfg.lookback_minutes));
    let range = since..now;

    tracing::debug!("Scanning range {range:?}");

    let excluder = std::sync::Arc::new(build_excluder()?);
    let finder = std::sync::Arc::new(linkify::LinkFinder::new());

    let channel_results = stream::iter(cfg.channel_ids.split(',').map(|ch_id| {
        let client = client.clone();
        let range = range.clone();
        let excluder = excluder.clone();
        let finder = finder.clone();
        let ch_id = ch_id.to_string();

        async move { fetch_channel_links(&client, &ch_id, range, &finder, &excluder).await }
    }))
    .buffer_unordered(8)
    .collect::<Vec<_>>()
    .await;

    let mut all_urls: Vec<String> = Vec::new();
    for result in channel_results {
        match result {
            Ok(urls) => all_urls.extend(urls),
            Err(e) => tracing::error!("channel fetch error: {e:?}"),
        }
    }

    if all_urls.is_empty() {
        tracing::info!("No new links since {since}. Nothing to write.");
        return Ok(());
    }

    let output = all_urls.join("\n");

    match &cfg.output_path {
        Some(path) => {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            writeln!(file, "{output}")?;
            tracing::info!("Wrote {} URLs to {}", all_urls.len(), path.display());
        }
        None => println!("{output}"),
    }

    Ok(())
}

#[tracing::instrument(skip(client, range, finder, excluder))]
async fn fetch_channel_links<H: HttpClient>(
    client: &DiscordClient<H>,
    ch_id: &str,
    range: impl std::ops::RangeBounds<UtcDateTime> + Clone,
    finder: &linkify::LinkFinder,
    excluder: &aho_corasick::AhoCorasick,
) -> Result<Vec<String>> {
    let ch = client.get_channel(ch_id).await?;
    let guild_id = ch
        .guild_id
        .ok_or_else(|| Error::MissingGuildId(ch_id.to_string()))?;
    let server_name = client.get_guild(&guild_id).await?.name;

    let messages = client.get_messages_range(ch_id, range, None).await?;

    if let Some(m) = messages.first() {
        tracing::debug!(
            ts = m
                .timestamp()?
                .format(&time::format_description::well_known::Rfc3339)?,
            snip = &m.content,
            "first message in range",
        );
    }

    let all_links: Vec<String> = messages
        .into_iter()
        .flat_map(|m| {
            finder
                .links(&m.content)
                .map(|l| l.as_str().to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    let excluded = all_links.iter().filter(|u| excluder.is_match(u)).count();
    let kept: Vec<String> = all_links
        .into_iter()
        .filter(|u| !excluder.is_match(u))
        .collect();

    tracing::info!(
        channel = ch.name,
        server = server_name,
        kept = kept.len(),
        excluded,
        "link extraction complete",
    );

    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_limit() {
        // We need a dummy client to test this, but we can also test utils directly
        let res = utils::parse_month("2023-13");
        assert!(matches!(res, Err(Error::MonthOutOfRange(13))));
    }

    #[test]
    fn test_invalid_snowflake() {
        let res = utils::snowflake_to_unix_ms("not-a-snowflake");
        assert!(matches!(res, Err(Error::InvalidSnowflake(_, _))));
    }

    #[test]
    fn test_config_missing_token() {
        unsafe { std::env::remove_var("DISCORD_TOKEN") };
        let res = Config::from_env();
        assert!(matches!(res, Err(Error::MissingToken)));
    }

    #[test]
    fn test_error_variants() {
        let err = Error::HttpStatus(404);
        assert_eq!(err.to_string(), "HTTP status 404");

        let err = Error::RateLimited {
            retry_after: Duration::from_secs(5),
        };
        assert_eq!(err.to_string(), "rate limited, retry after 5s");
    }
}
