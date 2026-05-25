/*!
llm-message-dedup: remove duplicate messages from LLM conversation histories.

Two messages are considered duplicate when they have the same role and
identical content (exact string or JSON match). Adjacent-only mode removes
consecutive duplicates only; global mode removes all repeated messages.

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
*/

use serde_json::Value;

fn message_key(msg: &Value) -> String {
    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
    let content = match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    format!("{}:{}", role, content)
}

/// Remove consecutive duplicate messages (same role + content).
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

/// Remove all duplicate messages, keeping first occurrence.
pub fn dedup_global(messages: Vec<Value>) -> Vec<Value> {
    let mut seen = std::collections::HashSet::new();
    messages.into_iter().filter(|msg| seen.insert(message_key(msg))).collect()
}

/// Remove messages where content is empty or whitespace-only.
pub fn remove_empty(messages: Vec<Value>) -> Vec<Value> {
    messages.into_iter().filter(|msg| {
        match msg.get("content") {
            Some(Value::String(s)) => !s.trim().is_empty(),
            Some(Value::Null) | None => false,
            _ => true,
        }
    }).collect()
}

/// Count how many duplicate messages exist (total - unique).
pub fn duplicate_count(messages: &[Value]) -> usize {
    let mut seen = std::collections::HashSet::new();
    let mut dups = 0;
    for msg in messages {
        let key = message_key(msg);
        if !seen.insert(key) {
            dups += 1;
        }
    }
    dups
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn u(s: &str) -> Value { json!({"role": "user", "content": s}) }
    fn a(s: &str) -> Value { json!({"role": "assistant", "content": s}) }

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
        let msgs = vec![u("hello"), json!({"role": "user", "content": "   "}), a("ok")];
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
}
