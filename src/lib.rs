#![allow(dead_code)]
use std::future::Future;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime as UtcDateTime;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// A Discord Snowflake (ID). Transmitted as a string in JSON, but 64-bit internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Snowflake(pub u64);

impl serde::Serialize for Snowflake {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for Snowflake {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse::<u64>()
            .map(Snowflake)
            .map_err(serde::de::Error::custom)
    }
}

impl std::str::FromStr for Snowflake {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Self(s.parse()?))
    }
}

impl std::fmt::Display for Snowflake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

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

    pub async fn get_channel(&self, channel_id: Snowflake) -> Result<Channel> {
        self.get_json(&format!("/channels/{channel_id}")).await
    }

    pub async fn get_guild(&self, guild_id: Snowflake) -> Result<Guild> {
        self.get_json(&format!("/guilds/{guild_id}")).await
    }

    pub async fn get_messages(&self, channel_id: Snowflake, limit: u8) -> Result<Vec<Message>> {
        if limit == 0 {
            return Err(Error::InvalidLimit);
        }
        self.get_json(&format!("/channels/{channel_id}/messages?limit={limit}"))
            .await
    }

    pub async fn get_messages_before(
        &self,
        channel_id: Snowflake,
        before_id: Snowflake,
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
        channel_id: Snowflake,
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
                self.get_messages_before(channel_id, before_id, batch_size(0))
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
            let last_id = last.id;

            let batch = filter_messages(
                self.get_messages_before(channel_id, last_id, batch_size(messages.len()))
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Channel {
    pub id: Snowflake,
    #[serde(rename = "type")]
    pub channel_type: u8,
    pub name: Option<String>,
    pub guild_id: Option<Snowflake>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Guild {
    pub id: Snowflake,
    pub name: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct User {
    pub id: Snowflake,
    pub username: String,
    pub global_name: Option<String>,
    #[serde(default)]
    pub bot: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Message {
    pub id: Snowflake,
    pub channel_id: Snowflake,
    pub guild_id: Option<Snowflake>,
    pub author: User,
    pub content: String,
    #[serde(with = "time::serde::rfc3339")]
    pub timestamp: UtcDateTime,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    #[serde(default)]
    pub embeds: Vec<Embed>,
    pub message_reference: Option<MessageReference>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Attachment {
    pub id: Snowflake,
    pub filename: String,
    pub url: String,
    pub content_type: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Embed {
    pub url: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MessageReference {
    pub message_id: Option<Snowflake>,
    pub channel_id: Option<Snowflake>,
    pub guild_id: Option<Snowflake>,
}

impl Message {
    pub fn timestamp(&self) -> Result<UtcDateTime> {
        utils::snowflake_to_utc_datetime(self.id)
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

pub mod utils {
    use crate::{Error, Result, Snowflake, UtcDateTime};
    use time::{Date, Month};

    const DISCORD_EPOCH: i64 = 1_420_070_400_000;

    pub fn snowflake_to_unix_ms(snowflake: Snowflake) -> Result<i64> {
        Ok((snowflake.0 >> 22) as i64 + DISCORD_EPOCH)
    }

    pub fn snowflake_to_utc_datetime(snowflake: Snowflake) -> Result<UtcDateTime> {
        let ms = snowflake_to_unix_ms(snowflake)?;
        UtcDateTime::from_unix_timestamp(ms / 1000)
            .map_err(|e| Error::InvalidTimestamp(e.to_string()))
    }

    pub fn unix_ms_to_snowflake(
        timestamp_ms: i64,
        worker_id: u16,
        sequence: u16,
    ) -> Result<Snowflake> {
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
        Ok(Snowflake(snowflake))
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
    let now = UtcDateTime::now_utc();
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
        let ch_id = ch_id.parse::<Snowflake>();

        async move {
            let ch_id = ch_id.map_err(|e| Error::InvalidSnowflake("".to_string(), e))?;
            fetch_channel_links(&client, ch_id, range, &finder, &excluder).await
        }
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
    ch_id: Snowflake,
    range: impl std::ops::RangeBounds<UtcDateTime> + Clone,
    finder: &linkify::LinkFinder,
    excluder: &aho_corasick::AhoCorasick,
) -> Result<Vec<String>> {
    let ch = client.get_channel(ch_id).await?;
    let guild_id = ch
        .guild_id
        .ok_or_else(|| Error::MissingGuildId(ch_id.to_string()))?;
    let server_name = client.get_guild(guild_id).await?.name;

    let mut messages = client.get_messages_range(ch_id, range, None).await?;
    for m in &mut messages {
        m.guild_id = Some(guild_id);
    }

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
            let mut links = finder
                .links(&m.content)
                .map(|l| l.as_str().to_string())
                .collect::<Vec<_>>();

            for attachment in m.attachments {
                links.push(attachment.url);
            }

            for embed in m.embeds {
                if let Some(url) = embed.url {
                    links.push(url);
                }
            }

            links
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
    fn test_invalid_snowflake_parse() {
        let res = "not-a-snowflake".parse::<Snowflake>();
        assert!(res.is_err());
    }

    #[test]
    fn test_message_deserialization() {
        let json = r#"{
            "id": "1197179023961378856",
            "channel_id": "832407655440056381",
            "content": "Check this out!",
            "timestamp": "2024-01-17T14:30:00Z",
            "author": {
                "id": "123456789",
                "username": "testuser",
                "global_name": "Test User",
                "bot": false
            },
            "attachments": [
                {
                    "id": "987654321",
                    "filename": "image.png",
                    "url": "https://example.com/image.png",
                    "content_type": "image/png"
                }
            ],
            "embeds": [
                {
                    "url": "https://example.com",
                    "title": "Example",
                    "description": "An example embed"
                }
            ]
        }"#;

        let msg: Message = serde_json::from_str(json).unwrap();
        assert_eq!(msg.id.0, 1197179023961378856);
        assert_eq!(msg.author.username, "testuser");
        assert_eq!(msg.attachments.len(), 1);
        assert_eq!(msg.embeds.len(), 1);
        assert_eq!(msg.attachments[0].filename, "image.png");
    }
    #[test]
    fn test_snowflake_roundtrip() {
        let s = "1197179023961378856";
        let sn: Snowflake = s.parse().unwrap();
        assert_eq!(sn.to_string(), s);
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
