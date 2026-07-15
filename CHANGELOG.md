# Changelog

All notable changes to Self-Evolution Skill TradingAgents are documented here.

The project follows a source-first release policy: generated artifacts, local
credential files, caches, and archives are excluded from version control.

## 0.3.0

### Added

- China A-share workflow support for continuous multi-agent trading runs.
- Skill-evolution modules for experience extraction, candidate skill synthesis,
  and verifier-gated acceptance.
- Walk-forward and weekly online controllers for no-lookahead skill iteration.
- CI coverage for Python 3.10, 3.11, 3.12, and 3.13.
- Strict Ruff linting for the public source tree.

### Changed

- Reworked the public README for a source-only open-source layout.
- Replaced upstream-facing CLI footer text and network user-agent strings with
  this project's public repository identity.
- Expanded ignore rules for generated artifacts, local caches, archives, and
  credential material.

### Removed

- Credential template files from the public tree.
- Upstream marketing images and unused top-level scratch files.
