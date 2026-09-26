# How this corpus was written

- **`vocab/*.json`:** kilix-needle's own name tables (`apps.py`, commit
  `96f2ed8`): the canonical ids of the Kilix catalog and kilix-settings
  controls, with the ways a request may say them.
- **`actions/*.json`:** written 2026-09-26 by three fresh agents from
  `TEMPLATE-SPEC.md` and the vocab files alone. They read no eval set, no
  research notes and no kilix-needle code.
  - launch and settings: 107 and 40 templates;
  - show, pane_stat and game: 114, 51 and 50;
  - compound, refuse and off-domain: 79, 125 and 51.
- **Checks.** kilix-needle's `apps.interpret` admits every generated row as
  labelled: 3,042 rows at seed 0, 6 per template. None of the 470 do-nothing
  rows admits anything under every plausible misreading. 26 rows equal to an
  eval request are dropped by `--exclude`.
- **No LLM-generated data at training time:** no Gemini, no DeepSeek, no
  OpenRouter.
