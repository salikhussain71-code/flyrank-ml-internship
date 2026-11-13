# Security and privacy

- The Hugging Face token lives only in your local `.env` (git-ignored). `.env.example` has a placeholder.
- Notebooks must have outputs cleared; a test fails if outputs are stored.
- Public outputs use pseudonymous page IDs (`P-xxxxxxxx`). No URLs, domains, client names or queries.
- No raw exports: `.parquet` and `.duckdb` are git-ignored and a test fails if any are present.
- A test scans the repo for token-like strings.
- If a token is ever exposed, revoke it at huggingface.co > Settings > Access Tokens and create a new one.
