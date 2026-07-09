# `templates/` — Scaffolds

> Copy-to-extend. These enable adding assets, specs, and experiments without reinventing structure.

| Template | Copy to | When |
|----------|---------|------|
| `asset-profile.example.yaml` | `config/assets/<asset_id>.yaml` | Onboarding a new tradeable asset (Gold, BTC…). See `../specs/assets/_onboarding-playbook.md`. |
| `spec-template.md` | `../specs/<domain>/<name>.md` or `../specs/tracks/<track>.md` | Writing a new reference spec or strategy-track spec. |
| `experiment-config-template.md` | `config/experiments/<exp_id>.yaml` | Running a new experiment. Follow `../rules/experiment-protocol.md`. |

> Do NOT put dense specs in `../rules/` (auto-loaded every session). Reference specs go in `../specs/`.
