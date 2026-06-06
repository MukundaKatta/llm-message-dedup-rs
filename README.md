# llm-message-dedup

A small, dependency-light Rust crate for removing duplicate, repeated, or empty
messages from LLM conversation histories.

When building agents or chat applications, message histories often accumulate
redundant entries — identical retries, repeated system prompts, or empty
placeholder messages. Trimming these before sending them back to a model saves
tokens and keeps context clean. This crate provides simple, predictable
functions to do exactly that.

Two messages are considered duplicates when they share the **same role** and
have **identical content** (compared as an exact string, or by canonical JSON
representation for non-string content).

## Features

- `dedup_adjacent` — remove only *consecutive* duplicate messages.
- `dedup_global` — remove *all* repeated messages, keeping the first occurrence.
- `remove_empty` — drop messages whose content is empty, whitespace-only, or null.
- `duplicate_count` — count how many messages are duplicates (total minus unique).

Messages are represented as [`serde_json::Value`] objects with `role` and
`content` fields, matching the common chat-message shape used by most LLM APIs.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llm-message-dedup = "0.1"
serde_json = "1"
```

## Usage

```rust
use llm_message_dedup::{dedup_adjacent, dedup_global, remove_empty, duplicate_count};
use serde_json::json;

let msgs = vec![
    json!({"role": "user", "content": "hello"}),
    json!({"role": "user", "content": "hello"}),
    json!({"role": "assistant", "content": "hi"}),
];

// Remove consecutive duplicates only.
let out = dedup_adjacent(msgs.clone());
assert_eq!(out.len(), 2);

// Remove all duplicates, keeping the first occurrence.
let out = dedup_global(msgs.clone());
assert_eq!(out.len(), 2);

// Count duplicates without modifying the input.
assert_eq!(duplicate_count(&msgs), 1);
```

### Adjacent vs. global

`dedup_adjacent` only collapses runs of identical neighbouring messages, so a
message that reappears later in the conversation is preserved:

```rust
use llm_message_dedup::dedup_adjacent;
use serde_json::json;

let msgs = vec![
    json!({"role": "user", "content": "hi"}),
    json!({"role": "assistant", "content": "ok"}),
    json!({"role": "user", "content": "hi"}),
];
// "hi" is not consecutive, so all three messages are kept.
assert_eq!(dedup_adjacent(msgs).len(), 3);
```

### Removing empty messages

```rust
use llm_message_dedup::remove_empty;
use serde_json::json;

let msgs = vec![
    json!({"role": "user", "content": "hello"}),
    json!({"role": "user", "content": "   "}),   // whitespace only -> removed
    json!({"role": "user", "content": null}),     // null -> removed
];
assert_eq!(remove_empty(msgs).len(), 1);
```

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** [`serde_json`](https://crates.io/crates/serde_json) only

## Development

```bash
cargo build
cargo test
```

The crate ships with a suite of unit tests covering adjacent and global
deduplication, empty-message removal, and duplicate counting.

## License

Licensed under the MIT License. See the `license` field in `Cargo.toml`.
