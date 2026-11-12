# Security notes

- API credentials are read from `OPENAI_API_KEY` and are never part of the repository.
- Do not commit proxy URLs, relay configurations, cluster paths, Slurm logs, model files or raw experiment outputs.
- If a credential is exposed, revoke it immediately and remove it from every Git revision before publishing.
