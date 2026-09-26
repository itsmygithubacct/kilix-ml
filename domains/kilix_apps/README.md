# kilix_apps

The training library for kilix-needle's `apps` job: launching Kilix apps and
games, and showing or hiding Kilix indicators, pane readouts, games and
settings sections. The job has five tools, in `tools.json`: `launch`, `show`,
`pane_stat`, `game` and `settings`.

`generate.py` fills the slot templates in `corpus/actions/` from
`corpus/vocab/` and writes `{query, actions, spans}` rows. `--exclude` drops
any row equal to an eval request.

`manifest.toml` is what `kilix-needle tune --job apps` reads. The environment
lock and its licence records are identical to kilix_panes'.

`corpus/blind_training_provenance.md` says how the corpus was written.
