//! Integration tests that exercise the public API exactly as an external
//! crate would (i.e. only through `pub` items).

use llm_message_dedup::{
    dedup, dedup_adjacent, dedup_global, duplicate_count, remove_empty, unique_count, DedupScope,
};
use serde_json::json;

#[test]
fn end_to_end_pipeline() {
    // A realistic, messy history: a repeated system prompt, a retried user
    // turn, a blank message, and a tool-calling assistant turn.
    let history = vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "What's the weather?"}),
        json!({"role": "user", "content": "What's the weather?"}),
        json!({"role": "user", "content": "   "}),
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "call_1"}]}),
        json!({"role": "tool", "content": "sunny", "tool_call_id": "call_1"}),
    ];

    // Strip blanks first, preserving the tool-call message despite null content.
    let cleaned = remove_empty(history);
    assert_eq!(
        cleaned.len(),
        6,
        "only the whitespace-only message is removed"
    );

    // Then drop the global duplicates (system prompt + user turn each appear twice).
    let deduped = dedup(cleaned, DedupScope::Global);
    let roles: Vec<&str> = deduped
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool"]);
}

#[test]
fn adjacent_vs_global_agree_on_pure_runs() {
    let msgs = vec![
        json!({"role": "user", "content": "a"}),
        json!({"role": "user", "content": "a"}),
        json!({"role": "user", "content": "b"}),
    ];
    assert_eq!(
        dedup_adjacent(msgs.clone()),
        dedup_global(msgs),
        "for purely consecutive duplicates the two scopes produce the same result"
    );
}

#[test]
fn counts_describe_the_input() {
    let msgs = [
        json!({"role": "user", "content": "x"}),
        json!({"role": "user", "content": "x"}),
        json!({"role": "assistant", "content": "y"}),
    ];
    assert_eq!(unique_count(&msgs), 2);
    assert_eq!(duplicate_count(&msgs), 1);
    assert_eq!(dedup_global(msgs.to_vec()).len(), unique_count(&msgs));
}

#[test]
fn dedup_scope_is_copy_and_comparable() {
    let scope = DedupScope::Global;
    let also = scope; // Copy
    assert_eq!(scope, also);
    assert_ne!(DedupScope::Adjacent, DedupScope::Global);
}

#[test]
fn content_type_is_significant() {
    // A string "5" and the number 5 are different JSON values and must not be
    // collapsed into one message.
    let msgs = vec![
        json!({"role": "user", "content": "5"}),
        json!({"role": "user", "content": 5}),
        json!({"role": "user", "content": "5"}), // exact repeat of the first
    ];
    assert_eq!(unique_count(&msgs), 2);
    assert_eq!(dedup_global(msgs).len(), 2);
}
