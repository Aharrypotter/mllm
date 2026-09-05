# jinja.cpp (vendored)

- Upstream: https://github.com/wangzhaode/jinja.cpp
- Upstream commit: a1d18d5979b3582a17883aac9c2bed2e26c7a578
- Vendored files: `jinja.hpp` (upstream root) and `ujson.hpp` (upstream
  `third_party/ujson.hpp`); `LICENSE` is the upstream license. The upstream
  rapidjson and nlohmann copies are not vendored: mllm supplies nlohmann/json
  from `third_party/json`.
- Local modifications: `patches/0001-transformers-compat.patch`, applied on
  top of the upstream commit above:
  - `loop.previtem`, `loop.nextitem`, `loop.revindex`, `loop.revindex0`;
  - an opt-in `JINJA_PRESERVE_JSON_OBJECT_ORDER` / `UJSON_USE_ORDERED_JSON`
    mode that keeps JSON object insertion order (Transformers `tojson`
    parity). In this mode a scope document is vector-backed, so
    `Context::set` and attribute assignment detach the assigned value before
    inserting; otherwise `{% set c = message.role %}` can dangle;
  - `{% set name %}...{% endset %}` block assignment;
  - `dict.items()`, `dict.keys()`, `dict.values()`, `dict.get(key, default)`
    method calls, and the `min` / `max` filters;
  - `{{ value }}` prints non-string values with Python `str()` semantics
    (`None`, `True`, dict/list reprs) like Jinja2;
  - list concatenation with `+` and namespace attribute updates that write
    back to the defining scope (both ported from MNN's jinja.cpp fork);
  - a C++14+ `std::make_unique` alias.

To resync: check out the upstream commit, copy the two headers, and re-apply
the patch with `patch -p1 -d third_party/jinja.cpp < patches/0001-transformers-compat.patch`.
