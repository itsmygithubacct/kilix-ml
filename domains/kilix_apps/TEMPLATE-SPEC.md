# Template spec for the kilix_apps training corpus (for template writers)

You write JSON template files for training a small on-device model that turns a
plain request into Kilix desktop actions. You may read ONLY the vocabulary files
in domains/kilix_apps/corpus/vocab/ (they list the surface forms each slot can
take) and this spec. Do not read, list or search anything else on this machine
(no eval sets, no research notes, no kilix-needle code).

File format (one file per assignment, UTF-8 JSON):
{"action": "<name>", "templates": [
  {"text": "open {app}", "actions": [["launch", {"app": "$app"}]]},
  {"text": "hide the {item} and the {item2}", "actions": [["show", {"item": "$item", "on": false}], ["show", {"item": "$item2", "on": false}]]},
  {"text": "install {app}", "actions": []}
]}
Slots in "text": {app} {game} {tool} {item} {stat} {section}, optionally numbered ({game2}) for a second, different value of the same kind. "$slot" in actions binds the slot's canonical value. Slot surface forms are bare nouns ("pdf viewer", "chess bash", "battery") — write "the {app}" where English wants an article; both "open the {app}" and "open {app}" are fine.

Tools (the only actions):
- launch {app}: open an app, game or tool. Use {app} (apps), {game} (games; "play {game}"), {tool} (launcher, temps, memory, mixer, transcripts).
- show {item, on}: show (true) or hide (false) a top-bar indicator or pane button ({item}).
- pane_stat {stat, mode}: {stat} is cpu or memory; mode is "auto", "always" or "off" — WRITE THE MODE LITERALLY in the text using one of these words, and put the canonical mode in the action: always: always / all the time / permanently / at all times / constantly; off: off / never / hide / hidden / disable / stop showing / no longer / don't need / anymore; auto: auto / automatic / automatically / when busy / only when busy / when needed / only when needed.
- game {game, available}: make a game available (true) or unavailable (false) in the Kilix games list — NOT playing it.
- settings {section}: open the Kilix settings screen at a section ({section}); the text must contain one of: settings, setting, preferences, options, configure, section, page, screen, panel; and must NOT say "center".

Rules the tool enforces (templates that break them are thrown away, so follow them):
- launch: an opening verb in the same clause as the name: open, launch, start, run, play, fire up, boot, bring up, pull up, load, show me, give me, get me, spin up, let's play, want to play, like to play, in the mood for, up for a round of, a game of. Not -ing forms ("playing", "opening").
- show and game: exactly one polarity in the clause. On: show, display, turn on / turn X on, switch on, put back / put X back, bring back, add back, enable, re-enable, unhide, restore, reveal, want to see, I want, I'd like, allow, unblock, add, put, make X available. Off: hide, remove, turn off / turn X off, switch off, get rid of, stop showing, don't show, no longer show, don't want, don't need, disable, drop, ditch, lose, block, take X off/away/out, make X unavailable. Never mix both in one clause; no other negation words in that clause.
- Clauses split at "and", "then", commas, "and then". A bare noun clause after a verb clause takes its verb ("hide the {item} and the {item2}"). "it" in the clause right after one naming a game/app refers to it ("enable {game} and then launch it").
- A request that reports someone else's words, installs/updates/removes/downloads, or is negated must do nothing.

Style: vary register (terse, polite, casual, indirect), sentence shape, word order, politeness wrappers ("could you", "please", "for me"), and verbs. No duplicates. Every template must have exactly one correct reading under these rules. Do not try to recall any existing dataset.
