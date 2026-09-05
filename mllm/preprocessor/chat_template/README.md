# Chat templates

This directory owns mllm's model-independent chat-template boundary. A model
converts its request into a `ChatTemplateRequest` (messages, tools, extra
template variables) and asks a `ChatPreprocessor` for the prompt text. The
model configuration selects the backend explicitly:

- `"chat_template_backend": "legacy"` (default while the rollout continues)
  keeps the model's migration renderer, a byte-stable copy of the prompt the
  model produced before this layer existed;
- `"chat_template_backend": "jinja_required"` renders the official template
  found in the model directory with the vendored `jinja.cpp` engine.

The build option `-DMLLM_ENABLE_JINJA_CHAT_TEMPLATE=ON` only controls whether
Jinja support is present in the binary. A `jinja_required` model fails during
loading when that support or its template is missing; it never silently falls
back to `legacy`.

## Pipeline

```text
runner / service request
   -> model request builder      (messages, content blocks, enable_thinking, tools)
   -> ChatPreprocessor.render    (legacy renderer | official Jinja template)
   -> model prompt adapter       (Qwen3.5: video placeholder -> timestamped frame markers)
   -> model tokenizer            (special tokens, BPE, image/video token expansion)
   -> token ids + media tensors
```

Migrated entry points:

| Model | Entry point | Backend source | Template discovery |
| --- | --- | --- | --- |
| Qwen3 service / probing | `Qwen3Session` | `config.json` | model directory |
| Qwen3.5 (text, image, video) | `Qwen3_5Tokenizer::convertMessage` | `Qwen3_5Config` | next to `tokenizer.json` |
| MiniCPM5 | `MiniCPM5Tokenizer::convertMessage` | `MiniCPM5Config` | next to `tokenizer.json` |
| Qwen3 runner, Qwen3-MoE, Qwen Ascend, MiniCPM4, Qwen NPU | `<Model>Tokenizer::convertMessage` via `LegacyChatMl.hpp` | model config | next to `tokenizer.json` |

The `legacy` renderers accept only the request shapes their runners have ever
produced (Qwen3.5: one user turn; MiniCPM5 and the ChatML runners: optional
system plus one user turn). Anything else fails closed so a missing template is never approximated.
The one intentional difference between `legacy` and the official templates is
that the official Qwen3.5 template trims message content; the parity gate uses
prompts without leading or trailing whitespace.

## Compatibility contract

- Hugging Face Transformers `apply_chat_template` is the semantic oracle.
- A reference case pins the model revision, template hash, Transformers/Jinja
  versions, rendered UTF-8 bytes, and tokenizer token IDs.
- Template rendering failures are errors. They must not silently fall back to
  ChatML, a raw prompt, or a model-specific formatter.
- `messages`, `tools`, and extra template variables retain JSON insertion
  order so `tojson` matches the reference environment.
- Special tokens that Transformers exposes as template variables
  (`bos_token`, `eos_token`, `pad_token`, `unk_token`, `sep_token`,
  `cls_token`, `mask_token`) are read from `tokenizer_config.json`; request
  `extra_context` overrides them.

Template discovery uses this order:

1. an explicit template path;
2. `<model directory>/chat_template.jinja` plus named templates in
   `<model directory>/additional_chat_templates/*.jinja`;
3. the legacy `chat_template` field in `tokenizer_config.json`, including the
   named-template array format.

When multiple templates exist, an explicitly selected name wins. Otherwise a
provided `tools` value selects `tool_use` at request time when available, then
`default` is used. Multiple templates without either selection fail closed,
matching Transformers.

The renderer only produces prompt text. Tokenization and special-token policy,
multimodal preprocessing, model state across turns, tool-call parsing, and
decode-time grammar constraints remain separate product stages.

## Engine

`third_party/jinja.cpp` is a vendored copy of `wangzhaode/jinja.cpp` plus the
patch recorded in its `UPSTREAM.md` (insertion-ordered objects, `loop`
neighbours, block assignment, mapping methods, `min`/`max`). Every engine gap
closed for an official template gets a guard in
`tests/preprocessor/ChatTemplateTest.cpp`.

## Parity gate

```bash
python3 tests/preprocessor/compare_transformers_chat_template.py \
  --model minicpm5 --model-dir <official MiniCPM5-1B> \
  --renderer <build>/bin/Mllm-ChatTemplate-Render \
  --probe <build>/bin/Mllm-ChatTemplate-Probe --config <mllm config.json>
```

Stage A compares rendered bytes, Transformers token IDs, and mllm tokenizer
IDs for pinned cases. Stage B runs the product path (`convertMessage`) with
both backends and compares the final token IDs with each other and with
Transformers.
