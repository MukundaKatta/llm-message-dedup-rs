# llm-message-dedup

[![CI](https://github.com/MukundaKatta/llm-message-dedup-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-message-dedup-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/llm-message-dedup.svg)](https://crates.io/crates/llm-message-dedup)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Remove duplicate or near-duplicate messages from LLM conversation history.

LLM agents and tooling tend to accumulate repeated messages: a prompt gets
retried, a client re-sends the same context, or two pipeline stages each append
the same system message. Those duplicates waste tokens and can confuse the
model. `llm-message-dedup` is a tiny, dependency-light crate (only
[`serde_json`](https://crates.io/crates/serde_json)) that cleans such histories.

## What counts as a duplicate?

Two messages are duplicates when they have the **same `role`** and **identical
`content`**:

- string content is compared by exact string match;
- structured content (e.g. an array of content blocks) is compared by its
  canonical JSON value.

So `{"role": "user", "content": "hi"}` and
`{"role": "assistant", "content": "hi"}` are **not** duplicates (different
roles), while two identical user turns are.

## Install

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llm-message-dedup = "0.1"
serde_json = "1"
```

or with cargo:

```sh
cargo add llm-message-dedup serde_json
```

## Usage

```rust
use llm_message_dedup::{dedup, remove_empty, DedupScope};
use serde_json::json;

fn main() {
    // A messy history: a repeated system prompt, a retried user turn,
    // a blank message, and a tool call.
    let history = vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "What's the weather?"}),
        json!({"role": "user", "content": "What's the weather?"}),
        json!({"role": "user", "content": "   "}),
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1"}]}),
        json!({"role": "tool", "content": "sunny", "tool_call_id": "call_1"}),
    ];

    // Drop blank messages (tool-call/tool-result messages are preserved even
    // though their textual content is empty).
    let cleaned = remove_empty(history);

    // Remove every repeated message, keeping the first occurrence.
    let deduped = dedup(cleaned, DedupScope::Global);

    let roles: Vec<&str> =
        deduped.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool"]);
}
```

## API

All functions operate on `serde_json::Value` messages, each expected to look
like `{"role": ..., "content": ...}`.

| Item | Description |
| --- | --- |
| `dedup(messages, scope)` | Remove repeats according to a [`DedupScope`]. Convenience wrapper over the two functions below. |
| `dedup_adjacent(messages)` | Remove only **consecutive** duplicates. Non-adjacent repeats are kept. |
| `dedup_global(messages)` | Remove **every** repeat, keeping the first occurrence. |
| `remove_empty(messages)` | Drop messages whose `content` is empty, whitespace-only, `null`, or absent — **unless** the message carries a tool payload (`tool_calls` array or `tool_call_id`), which is always kept. |
| `duplicate_count(messages)` | Number of repeated messages (`len - unique_count`). |
| `unique_count(messages)` | Number of distinct messages. |
| `DedupScope` | `Adjacent` or `Global`; selects how aggressively `dedup` removes repeats. |

### Adjacent vs. global

```rust
use llm_message_dedup::{dedup_adjacent, dedup_global};
use serde_json::json;

let msgs = vec![
    json!({"role": "user", "content": "hi"}),
    json!({"role": "assistant", "content": "ok"}),
    json!({"role": "user", "content": "hi"}), // a non-adjacent repeat
];

// Adjacent mode keeps the non-consecutive repeat.
assert_eq!(dedup_adjacent(msgs.clone()).len(), 3);
// Global mode removes it.
assert_eq!(dedup_global(msgs).len(), 2);
```

### Counting without mutating

```rust
use llm_message_dedup::{duplicate_count, unique_count};
use serde_json::json;

let msgs = [
    json!({"role": "user", "content": "hi"}),
    json!({"role": "user", "content": "hi"}),
    json!({"role": "assistant", "content": "ok"}),
];
assert_eq!(unique_count(&msgs), 2);
assert_eq!(duplicate_count(&msgs), 1);
```

## Notes and limitations

- Messages with no `role` or no `content` are handled gracefully (missing
  fields are treated as JSON `null` for keying purposes).
- Comparison is **type-aware**: the string content `"5"` and the number
  content `5` are treated as distinct messages, as are `"true"`/`true` and
  `"null"`/`null`. Identity keys are derived from the canonical JSON of the
  `role`/`content` pair, so distinct messages never collide.
- "Near-duplicate" matching is currently exact: there is no fuzzy/semantic
  comparison. Normalize your content (trim, lowercase, etc.) beforehand if you
  need looser matching.
- The crate never reorders messages; it only removes elements.

## Development

```sh
cargo build
cargo test          # unit + integration + doc tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## License

Licensed under the [MIT License](https://opensource.org/licenses/MIT).
