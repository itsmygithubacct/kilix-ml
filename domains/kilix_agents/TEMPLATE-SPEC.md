# Template spec for the kilix_agents training corpus (for template writers)

You write JSON template files for training a small on-device model that turns a
plain request into actions on coding-agent sessions (Claude Code, Codex, Grok,
Qwen via omp) in the Kilix terminal. Coding sessions themselves often make these
requests. You may read ONLY this spec and the files in
domains/kilix_agents/corpus/vocab/. Do not read, list or search anything else
on this machine: no eval sets, no research notes, no kilix-needle code.

File format (one file per assignment, UTF-8 JSON):
{"action": "<name>", "templates": [
  {"text": "open {agent} in {dir}", "actions": [["agent", {"agent": "$agent", "dir": "$dir"}]]},
  {"text": "spawn a {agent} reviewer in {dir}: {task}", "actions": [["agent", {"agent": "$agent", "dir": "$dir", "prompt": "$task"}]]},
  {"text": "wait until the {agent} session in {dir} is done", "actions": [["wait", {"session": "$agent@$dir", "for": "idle"}]]},
  {"text": "install {agent}", "actions": []}
]}

Slots in "text", each optionally numbered ({dir2}) for a second, different value:
- {agent}: claude, codex, grok, qwen-omp (surface forms in agents.json).
- {dir}: a directory as a person names it ("the needle repo", "~/src/website", "here").
- {place}: a side for a split, or a new tab (places.json maps them to right/left/down/up/tab).
- {model}: a model name ("opus", "gpt-6").
- {task}: a task for a new session, one line (tasks.json).
- {message}: a message to a session, one line (messages.json).
- {session}: a previous session's id or title, to resume (sessions.json).
In actions, "$slot" binds it. Session references for wait/tell are "$agent@$dir", or the literal "it" for the session the same request launched.

Tools (the only actions):
- agent {agent, dir, prompt?, resume?, model?, place?}: start a session. prompt = "$task" when the text gives a task; resume = "$session" when it asks to resume/continue one; model = "$model" when it names one; place = "$place" for a split or tab.
- wait {session, for, timeout?}: for = "idle" (done, finished, idle, complete, ready) or "waiting" (asks a question, needs approval, waiting, blocked). timeout (integer seconds) only when the text says a limit ("up to 10 minutes" gives 600).
- tell {session, text, wait?}: text = "$message"; wait = true only when the text says to wait until it is done first ("once it's idle, tell it ...").

Rules the tool enforces (templates that break them are thrown away, so follow them):
1. Outside the slots, use ONLY these words (plus digits and punctuation): a above after agent ahead also an and another approval as ask asking asks at away back be becomes below beside block blocked boot bring but can checkout cheers cli complete completed completes continue copy could create dir directory done down eight eleven fifteen fifty finish finished finishes fire five folder for forty forty-five four fresh from get gets give go goes half has have helper her hey his hold horizontal hour hours i idle in input instance instruct into is it it's its just kick kindly know launch left let let's lets limit longer max maximum me message min mins minute minutes model most my needs new next nine ninety notify now of off ok okay on once one open our over pane pick ping please pls project put question questions quick quickly reaches ready reopen repo repository resume review reviewer right run say saying saying: sec second seconds secs send separate session sessions seven side six sixty so spawn spin split start stop tab task tell telling ten than thank thanks that the them then there thirty this three thx till time timeout to too turn twelve twenty two ty until up us using vertical via wait waiting we when will window with work worker working workspace would you your
2. A launch needs one of: open, start, launch, spawn, fire up, spin up, run, bring up, kick off, get, boot, create, put; or it starts with the agent's name ("{agent} in {dir}, {task}"). Resume needs resume / continue / reopen / pick ... back.
3. Put payloads after a marker: "{agent} in {dir}: {task}", "... and tell it to {message}", "... to {task}". The payload is copied exactly; never add words inside it.
4. The words outside payloads must not negate (not, don't, never, no), report (said, told, someone, friend), take back (cancel, never mind, actually), ask a question, install/update, change permissions (yolo, skip approvals, bypass, sandbox), or close/kill/quit sessions. Those requests must do nothing: write them in the refuse assignment with "actions": [].
5. One sentence per request (a trailing "thanks" is fine).

Style: vary register (terse, casual, polite, and as an agent writes: "Spawn a codex reviewer in the needle repo: {task}"), word order, and the order of optional parts (model, place, prompt). No duplicates. Every template must have exactly one correct reading under these rules. Do not try to recall any existing dataset.
