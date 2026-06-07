/*!
llm-message-dedup: remove duplicate messages from LLM conversation histories.

LLM agents and tools frequently accumulate repeated messages in their
conversation history — for example when a prompt is retried, when a client
re-sends context, or when two pipeline stages append the same system message.
Those duplicates waste tokens and can confuse a model. This crate provides a
few small, dependency-light helpers to clean such histories.

Two messages are considered duplicate when they have the same `role` and
identical `content`. Content is compared by exact string match for string
content, and by canonical JSON value for structured content (e.g. arrays of
content blocks). Adjacent-only mode removes consecutive duplicates only;
global mode removes every repeated message, keeping the first occurrence.

```rust
use llm_message_dedup::{dedup_adjacent, dedup_global};
use serde_json::json;

let msgs = vec![
    json!({"role": "user", "content": "hello"}),
    json!({"role": "user", "content": "hello"}),
    json!({"role": "assistant", "content": "hi"}),
];
let out = dedup_adjacent(msgs);
assert_eq!(out.len(), 2);
```

A unified entry point is also available via [`dedup`] and [`DedupScope`]:

```rust
use llm_message_dedup::{dedup, DedupScope};
use serde_json::json;

let msgs = vec![
    json!({"role": "user", "content": "hi"}),
    json!({"role": "assistant", "content": "ok"}),
    json!({"role": "user", "content": "hi"}),
];
// Global scope removes the non-adjacent repeat.
assert_eq!(dedup(msgs, DedupScope::Global).len(), 2);
```
*/

use serde_json::Value;

/// Selects how aggressively [`dedup`] removes repeated messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DedupScope {
    /// Remove only consecutive duplicates (same as [`dedup_adjacent`]).
    Adjacent,
    /// Remove every repeated message, keeping the first occurrence
    /// (same as [`dedup_global`]).
    Global,
}

/// Build a canonical, collision-free identity key for a message.
///
/// The key is the JSON serialization of the two-element array
/// `[role, content]`. Encoding the pair as JSON (rather than concatenating the
/// two fields with a delimiter) avoids two classes of false matches:
///
/// * **Type collisions** — the string content `"5"` serializes to `"5"`
///   (with quotes) while the number content `5` serializes to `5` (without),
///   so they are no longer treated as the same message. The same holds for
///   `"true"`/`true` and `"null"`/`null`.
/// * **Delimiter collisions** — a literal-delimiter scheme such as
///   `format!("{role}:{content}")` makes `role="user", content="a:b"` and
///   `role="user:a", content="b"` indistinguishable. JSON encodes the field
///   boundaries explicitly, so distinct messages always map to distinct keys.
///
/// A missing `role` is treated as JSON `null`, and a missing `content` is
/// treated as JSON `null`, matching the previous "absent == empty" behavior
/// for keying purposes while keeping the two fields unambiguous.
fn message_key(msg: &Value) -> String {
    let role = msg.get("role").unwrap_or(&Value::Null);
    let content = msg.get("content").unwrap_or(&Value::Null);
    // serde_json serialization of a fixed-shape value is infallible.
    serde_json::to_string(&[role, content]).unwrap_or_default()
}

/// Remove repeated messages according to the given [`DedupScope`].
///
/// This is a convenience wrapper over [`dedup_adjacent`] and [`dedup_global`].
///
/// ```
/// use llm_message_dedup::{dedup, DedupScope};
/// use serde_json::json;
///
/// let msgs = vec![
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "user", "content": "hi"}),
/// ];
/// assert_eq!(dedup(msgs, DedupScope::Adjacent).len(), 1);
/// ```
pub fn dedup(messages: Vec<Value>, scope: DedupScope) -> Vec<Value> {
    match scope {
        DedupScope::Adjacent => dedup_adjacent(messages),
        DedupScope::Global => dedup_global(messages),
    }
}

/// Remove consecutive duplicate messages (same role + content).
///
/// Non-adjacent repeats are preserved; use [`dedup_global`] to remove those.
///
/// ```
/// use llm_message_dedup::dedup_adjacent;
/// use serde_json::json;
///
/// let msgs = vec![
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "assistant", "content": "yo"}),
/// ];
/// assert_eq!(dedup_adjacent(msgs).len(), 2);
/// ```
pub fn dedup_adjacent(messages: Vec<Value>) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::new();
    let mut last_key: Option<String> = None;
    for msg in messages {
        let key = message_key(&msg);
        if Some(&key) != last_key.as_ref() {
            last_key = Some(key);
            result.push(msg);
        }
    }
    result
}

/// Remove all duplicate messages, keeping the first occurrence.
///
/// ```
/// use llm_message_dedup::dedup_global;
/// use serde_json::json;
///
/// let msgs = vec![
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "assistant", "content": "ok"}),
///     json!({"role": "user", "content": "hi"}),
/// ];
/// assert_eq!(dedup_global(msgs).len(), 2);
/// ```
pub fn dedup_global(messages: Vec<Value>) -> Vec<Value> {
    let mut seen = std::collections::HashSet::new();
    messages
        .into_iter()
        .filter(|msg| seen.insert(message_key(msg)))
        .collect()
}

/// Returns `true` if a message carries tool calls, in which case it is
/// meaningful even when its textual `content` is absent or empty.
///
/// LLM tool-calling formats (OpenAI `tool_calls`, Anthropic `tool_use`
/// content blocks) routinely emit assistant messages whose `content` is
/// `null` or `""` while the actual payload lives elsewhere. Treating those as
/// empty would silently drop tool invocations from the history.
fn has_tool_payload(msg: &Value) -> bool {
    match msg.get("tool_calls") {
        Some(Value::Array(calls)) if !calls.is_empty() => return true,
        _ => {}
    }
    matches!(msg.get("tool_call_id"), Some(Value::String(s)) if !s.is_empty())
}

/// Remove messages whose `content` is empty or whitespace-only.
///
/// A message is dropped when its `content` is an empty string, a
/// whitespace-only string, JSON `null`, or absent — *unless* the message
/// carries a tool payload (a `tool_calls` array or a `tool_call_id`), which
/// is preserved so tool invocations and their results are never lost.
/// Non-string, non-null content (for example an array of content blocks) is
/// always kept.
///
/// ```
/// use llm_message_dedup::remove_empty;
/// use serde_json::json;
///
/// let msgs = vec![
///     json!({"role": "user", "content": "hello"}),
///     json!({"role": "user", "content": "   "}),
///     json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}]}),
/// ];
/// let out = remove_empty(msgs);
/// // The blank user message is removed; the tool-call message is kept.
/// assert_eq!(out.len(), 2);
/// ```
pub fn remove_empty(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .filter(|msg| match msg.get("content") {
            Some(Value::String(s)) => !s.trim().is_empty() || has_tool_payload(msg),
            Some(Value::Null) | None => has_tool_payload(msg),
            _ => true,
        })
        .collect()
}

/// Count how many duplicate messages exist (total minus unique).
///
/// This counts repeats only; the first occurrence of each distinct message is
/// not counted. It equals `messages.len() - unique_count(messages)`.
///
/// ```
/// use llm_message_dedup::duplicate_count;
/// use serde_json::json;
///
/// let msgs = [
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "user", "content": "hi"}),
/// ];
/// assert_eq!(duplicate_count(&msgs), 2);
/// ```
pub fn duplicate_count(messages: &[Value]) -> usize {
    messages.len() - unique_count(messages)
}

/// Count how many distinct messages exist (by role + content key).
///
/// ```
/// use llm_message_dedup::unique_count;
/// use serde_json::json;
///
/// let msgs = [
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "user", "content": "hi"}),
///     json!({"role": "assistant", "content": "ok"}),
/// ];
/// assert_eq!(unique_count(&msgs), 2);
/// ```
pub fn unique_count(messages: &[Value]) -> usize {
    let mut seen = std::collections::HashSet::new();
    messages
        .iter()
        .filter(|msg| seen.insert(message_key(msg)))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn u(s: &str) -> Value {
        json!({"role": "user", "content": s})
    }
    fn a(s: &str) -> Value {
        json!({"role": "assistant", "content": s})
    }

    #[test]
    fn dedup_adjacent_removes_consecutive() {
        let msgs = vec![u("hi"), u("hi"), a("hello")];
        let out = dedup_adjacent(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn dedup_adjacent_keeps_non_consecutive_dups() {
        let msgs = vec![u("hi"), a("ok"), u("hi")];
        let out = dedup_adjacent(msgs);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn dedup_adjacent_empty() {
        assert!(dedup_adjacent(vec![]).is_empty());
    }

    #[test]
    fn dedup_adjacent_no_dups() {
        let msgs = vec![u("a"), a("b"), u("c")];
        let out = dedup_adjacent(msgs.clone());
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn dedup_global_removes_all_dups() {
        let msgs = vec![u("hi"), a("ok"), u("hi")];
        let out = dedup_global(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn dedup_global_keeps_first() {
        let msgs = vec![u("first"), u("second"), u("first")];
        let out = dedup_global(msgs);
        assert_eq!(out[0]["content"], "first");
        assert_eq!(out[1]["content"], "second");
    }

    #[test]
    fn different_roles_not_deduped() {
        let msgs = vec![
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let out = dedup_adjacent(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn remove_empty_strips_blank_content() {
        let msgs = vec![
            u("hello"),
            json!({"role": "user", "content": "   "}),
            a("ok"),
        ];
        let out = remove_empty(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn remove_empty_keeps_non_empty() {
        let msgs = vec![u("a"), a("b")];
        let out = remove_empty(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn remove_empty_null_content() {
        let msgs = vec![json!({"role": "user", "content": null}), u("ok")];
        let out = remove_empty(msgs);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn duplicate_count_zero() {
        assert_eq!(duplicate_count(&[u("a"), a("b")]), 0);
    }

    #[test]
    fn duplicate_count_counts_extras() {
        let msgs = [u("hi"), u("hi"), u("hi")];
        assert_eq!(duplicate_count(&msgs), 2);
    }

    #[test]
    fn triple_consecutive_dedup_adjacent() {
        let msgs = vec![u("x"), u("x"), u("x"), a("y")];
        let out = dedup_adjacent(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn dedup_dispatches_to_adjacent() {
        let msgs = vec![u("hi"), a("ok"), u("hi")];
        // Adjacent scope keeps the non-adjacent repeat.
        assert_eq!(dedup(msgs, DedupScope::Adjacent).len(), 3);
    }

    #[test]
    fn dedup_dispatches_to_global() {
        let msgs = vec![u("hi"), a("ok"), u("hi")];
        // Global scope removes the non-adjacent repeat.
        assert_eq!(dedup(msgs, DedupScope::Global).len(), 2);
    }

    #[test]
    fn dedup_global_matches_structured_content() {
        let blocks = json!({"role": "user", "content": [{"type": "text", "text": "hi"}]});
        let msgs = vec![blocks.clone(), blocks];
        assert_eq!(dedup_global(msgs).len(), 1);
    }

    #[test]
    fn unique_count_counts_distinct() {
        let msgs = [u("a"), u("a"), a("b")];
        assert_eq!(unique_count(&msgs), 2);
    }

    #[test]
    fn unique_and_duplicate_counts_are_complementary() {
        let msgs = [u("a"), u("a"), u("a"), a("b")];
        assert_eq!(unique_count(&msgs) + duplicate_count(&msgs), msgs.len());
    }

    #[test]
    fn remove_empty_keeps_tool_call_message_with_null_content() {
        let msgs = vec![
            u("hello"),
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "c1"}]}),
        ];
        let out = remove_empty(msgs);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn remove_empty_keeps_tool_result_message() {
        let msgs = vec![
            json!({"role": "tool", "content": "", "tool_call_id": "c1"}),
            json!({"role": "user", "content": "   "}),
        ];
        let out = remove_empty(msgs);
        // Tool result is kept (has tool_call_id); blank user message is dropped.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "tool");
    }

    #[test]
    fn remove_empty_drops_empty_tool_calls_array() {
        let msgs = vec![json!({"role": "assistant", "content": null, "tool_calls": []})];
        // An empty tool_calls array is not a real payload, so it is dropped.
        assert!(remove_empty(msgs).is_empty());
    }

    #[test]
    fn string_and_number_content_are_distinct() {
        // Regression: previously both keyed to "user:5" and collapsed into one.
        let msgs = vec![
            json!({"role": "user", "content": "5"}),
            json!({"role": "user", "content": 5}),
        ];
        assert_eq!(dedup_global(msgs.clone()).len(), 2);
        assert_eq!(duplicate_count(&msgs), 0);
    }

    #[test]
    fn string_and_bool_content_are_distinct() {
        let msgs = vec![
            json!({"role": "user", "content": "true"}),
            json!({"role": "user", "content": true}),
        ];
        assert_eq!(dedup_global(msgs).len(), 2);
    }

    #[test]
    fn delimiter_collision_is_avoided() {
        // Regression: `format!("{role}:{content}")` made these two distinct
        // messages share the key "user:a:b".
        let msgs = vec![
            json!({"role": "user", "content": "a:b"}),
            json!({"role": "user:a", "content": "b"}),
        ];
        assert_eq!(dedup_global(msgs).len(), 2);
    }

    #[test]
    fn identical_string_content_still_deduped() {
        // The fix must not regress the ordinary case.
        let msgs = vec![u("hi"), u("hi")];
        assert_eq!(dedup_global(msgs).len(), 1);
    }

    #[test]
    fn identical_structured_content_still_deduped() {
        let blocks = json!({"role": "user", "content": [{"type": "text", "text": "hi"}]});
        let msgs = vec![blocks.clone(), blocks];
        assert_eq!(dedup_global(msgs).len(), 1);
    }
}
