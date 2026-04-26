# Module H verdict adversarial suite

The six verdicts Module H can emit each have a targeted end-to-end test in
`tests/test_module_h_verdicts.py`. Each test seeds a small (memory,
corpus) pair designed so the verdict is the only honest answer for that
pair, then asserts Module H produces it.

| Verdict | Adversarial setup |
|---|---|
| `well-supported`     | Memory claim has a near-verbatim anchor in the corpus. |
| `weakly-supported`   | Anchor exists but is partial / inferential; classifier returns `WEAKLY_SUPPORTED`. |
| `unsupported`        | Memory references content semantically nearby (so the similarity short-circuit doesn't fire) but the actual anchor doesn't back the claim; classifier returns `UNSUPPORTED`. |
| `contradicted`       | Memory and corpus are about the same topic but disagree on facts; classifier returns `CONTRADICTED`. |
| `insufficient-data`  | Memory claim has no semantic anchor in the corpus; top-1 cosine similarity below `INSUFFICIENT_THRESHOLD` short-circuits the classifier. |
| `out-of-scope`       | Memory is scoped to a `project_memories.<uuid>` entry whose project has no conversations in the audit sample (project attribution misses). |

The tests mock the Anthropic client and use a `StaticEmbeddingProvider`
so retrieval and classification outcomes are deterministic. They probe
the **plumbing** for each verdict — that the right code path fires and
the right `Finding` is constructed. They do **not** probe the classifier's
decision boundary; that requires calibration against held-out labelled
examples (see `docs/calibration.md`).

This README is the "fixture data" pointer in the project README's
`Honest limitations` section. Running:

```
uv run pytest tests/test_module_h_verdicts.py -v
```

exercises one test per verdict and confirms all six paths reach a Finding.
