# MeowSense Collection Bot — Design Document

**Status:** design only. No bot code exists yet, and this document does not
authorize a deployment. It is the plan a maintainer would implement *after* the
[open decisions](#10-open-decisions-for-the-maintainer) at the end are made.

**Why this is a document and not code.** The bot's hard parts are not technical.
Receiving audio and running a model is an afternoon; doing so while handling
consent, deletion, background human speech and the licensing of other people's
recordings responsibly is the actual work. Getting those wrong is worse than
having no bot, so they are specified before a line is written.

This expands [workstream 1 in the roadmap](ROADMAP.md#1-telegram-collection-bot).
Read that first for the *why*; this is the *how*.

---

## 1. Purpose and scope

**Goal.** Grow the training data from 21 cats to hundreds, with labels that
carry the situational context the audio alone cannot recover — directly
attacking the two limits the benchmark keeps hitting: too few cats, and `food`
being a context rather than a sound.

**In scope**
- Collect short cat vocalizations from owners via Telegram, with consent.
- Attach owner-supplied labels and light metadata (crucially a per-owner cat ID).
- Give the owner an immediate model guess, as the incentive to contribute.
- Produce a dataset that can be pooled with CatMeows (CC BY 4.0, same schema).

**Out of scope (deliberately)**
- A "cat translator" product. The honest accuracy is ~0.50–0.60 on unseen cats;
  the bot's guess is framed as a game, never as a translation. See
  [Non-goals in the roadmap](ROADMAP.md#non-goals).
- Real-time or on-device inference.
- Storing anything about the *human* beyond an opaque owner ID and their consent
  record.
- Medical use. If `pain`/illness labels are ever collected, that is a separate,
  vet-supervised effort — not this bot.

---

## 2. Interaction flow

The three-button loop from the roadmap, specified end to end. Every arrow that
stores data is marked ⬇.

```
/start
  └─ consent screen  ──[decline]──▶ nothing stored, bot explains and stops
       │
     [I agree]  ⬇ store consent record (version, timestamp, opaque owner id)
       │
       ▼
  one-time onboarding (skippable): "Which cat is this?" → cat label
       │
       ▼
  owner sends a voice note / audio clip
       │
       ├─ VAD / energy gate ──[no vocalization]──▶ "I couldn't hear a meow —
       │                                            try again?"  (nothing stored)
       │
     [passes]  ⬇ store ORIGINAL audio + metadata (status = pending)
       │
       ▼
  bot runs the current best model
       │
       ▼
  "I think this is: ASKING FOR FOOD (54%)"
  [ 🎯 guessed ]   [ ✏️ wrong ]   [ 🤷 don't know ]
       │                │                │
       │                │                └─ ⬇ label = unknown (semi-supervised pool)
       │                │
       │                └─ "What was it?" → context menu ─ ⬇ label = owner's choice
       │
       └─ ⬇ label = model prediction (confirmed by owner)
       │
       ▼
  optional one-tap enrichers (each skippable):
     "How long since food?"  ·  "Was someone playing with them?"
       │  ⬇ update metadata
       ▼
  "Thanks! That's clip #7 from Barsik. Send another any time."
```

**Design rules baked into the flow**
- **"Don't know" is a first-class answer.** Forcing a label manufactures noise.
- **The guess comes *before* the label prompt** so the owner reacts to it (one
  tap to confirm) instead of composing an answer from scratch — but the model's
  guess is never silently adopted; the owner's tap is what creates the label.
- **The original audio is stored, always**, before any processing. Downstream
  cleaning works on copies.
- **Nothing is stored before consent, and nothing that fails the VAD gate is
  stored at all.**

---

## 3. Data model

Three record types. IDs are opaque and internal; no Telegram username, phone
number or display name is ever persisted.

### 3.1 Owner

```jsonc
{
  "owner_id": "o_7f3a…",          // HMAC(telegram_user_id, server_secret), not reversible to the account
  "consent_version": "2026-07-01",
  "consent_at": "2026-07-23T14:02:11Z",
  "consent_scope": ["store", "train", "publish_cc_by_4_0"],
  "locale": "ru",                  // for choosing message language only
  "created_at": "2026-07-23T14:02:11Z",
  "status": "active"               // active | deletion_requested | deleted
}
```

`owner_id` is a keyed hash of the Telegram user id, so the same person maps to
the same owner across sessions **without** the database storing anything that
identifies the account. The mapping is one-way: given the database, you cannot
recover who anyone is.

### 3.2 Cat

```jsonc
{
  "cat_id": "o_7f3a…::c_01",       // namespaced under the owner; globally unique
  "owner_id": "o_7f3a…",
  "label": "Barsik",               // owner-facing only; never published
  "breed": "unknown",              // optional; free choice from a menu
  "sex": "unknown",                // optional: female/male × intact/neutered/unknown
  "age_bucket": "unknown",         // optional: kitten / adult / senior
  "created_at": "…"
}
```

**`cat_id` is mandatory for every clip.** Without it there is no grouped
validation, and the entire dataset inherits the leakage the project exists to
avoid. Onboarding asks for it once; if an owner refuses, their clips are still
stored but flagged `cat_id = unassigned` and **excluded from any released
dataset** until assigned. This is stated to the owner.

### 3.3 Clip

```jsonc
{
  "clip_id": "clip_9a2c…",
  "owner_id": "o_7f3a…",
  "cat_id": "o_7f3a…::c_01",
  "audio_uri": "s3://…/clip_9a2c….ogg",   // ORIGINAL, immutable
  "received_at": "2026-07-23T14:05:44Z",
  "duration_s": 2.3,
  "source_format": "ogg/opus",             // Telegram voice notes are opus

  // Label
  "label": "food",                         // food | play | greeting | distress | litter | vet | other | unknown
  "label_source": "corrected",             // model_confirmed | corrected | unknown
  "model_pred": "food",
  "model_confidence": 0.54,
  "model_version": "ast-svm-0.2.0",

  // Optional enrichers (each null unless the owner tapped)
  "hours_since_food": "1-3",               // bucketed
  "human_interacting": true,

  // Quality (server-computed, never shown to the owner)
  "vad_passed": true,
  "snr_estimate_db": 11.2,
  "flags": ["label_model_disagree_highconf"],

  "consent_version": "2026-07-01",         // snapshot at time of contribution
  "status": "active"                       // active | deletion_requested | deleted
}
```

### 3.4 Mapping to the CatMeows schema

The acoustic pipeline (`src/features.py`) reads context, cat id, breed, sex and
owner from CatMeows filenames. A published export renders each clip to the same
convention so the two datasets pool without special-casing:

```
<context>_<cat_id>_<breed>_<sex>_<owner_id>_<session><counter>.wav
```

with `context ∈ {B,F,I,…}` extended for the new classes, `unknown` fields left
as an explicit token, and audio transcoded opus→wav at the original rate. The
export is a separate, reviewed step — the live store keeps richer JSON.

---

## 4. Storage, retention and deletion

### 4.1 Where things live
- **Audio**: object storage (e.g. S3-compatible), private bucket, server-side
  encrypted, no public ACLs, no directory listing.
- **Metadata**: a small relational database (SQLite is enough to start;
  Postgres if it grows). Never store audio bytes in the database.
- **Secrets** (bot token, `server_secret`, storage keys): environment variables
  or a secret manager. **Never in the repo, never in logs.** See [Security](#7-security).

### 4.2 Retention
- Original audio is kept as long as the owner's consent stands and no deletion
  is requested.
- Raw Telegram update payloads (which contain account identifiers) are processed
  in memory and **not** persisted; only the derived, de-identified records above
  are written.
- Logs never contain audio, `telegram_user_id`, or message text — only opaque
  ids and event types.

### 4.3 Deletion — `/delete_my_data`

This must actually work, not tombstone a row.

```
/delete_my_data
  └─ confirm screen: "This permanently deletes every clip and label from
     all your cats, and cannot be undone. Delete?"  [Yes, delete]  [Cancel]
        │
      [Yes]
        ├─ hard-delete every audio object for this owner from storage
        ├─ hard-delete clip + cat rows
        ├─ mark owner status=deleted, drop all fields except owner_id + a
        │  deletion tombstone (so a re-contact does not silently resurrect data)
        └─ confirm: "Done. Everything is gone. You can /start again fresh."
```

- Deletion propagates to any **unpublished** export staging.
- Clips already included in a **published, versioned** dataset release cannot be
  retracted from third parties who downloaded it — this limit is stated plainly
  in the consent screen (§5). Future releases exclude the deleted data.
- A single-clip undo ("delete that last one") is a nice-to-have; whole-owner
  deletion is the guarantee.

---

## 5. Consent

Shown in full at `/start`, before anything is stored, in the owner's language,
plain words, no dark patterns. Illustrative copy (a real deployment must have
this reviewed — see [open decisions](#10-open-decisions-for-the-maintainer)):

> **Before we start.** MeowSense is an open research project about cat sounds.
> If you agree:
> - I store the cat sounds you send and the labels you give them.
> - I use them to train open models, and I may include them in a **public,
>   openly-licensed dataset (CC BY 4.0)** that anyone can download.
> - I do **not** store your name, username or phone number — only an internal
>   code for your account.
> - Recordings can pick up **human voices in the background.** Send only clips
>   you're comfortable releasing publicly, and avoid recording other people.
> - You can delete everything any time with /delete_my_data. Once a clip is in a
>   published dataset release, people who already downloaded it keep their copy —
>   I just stop including it going forward.
>
> [ I agree ]   [ No thanks ]

**Rules**
- Opt-in only. No pre-ticked boxes; "No thanks" ends the session cleanly.
- Consent is **versioned**. Changing what is collected requires re-consent
  before the new collection applies; old clips keep the version they were given
  under.
- A `/privacy` command reprints this any time.
- **Minors:** the bot is for adult owners recording their own pets. It cannot
  verify age, so it does not knowingly collect from minors and the consent text
  states the service is for adults — a real deployment must check its
  jurisdiction's rules here.

---

## 6. Data quality

Crowdsourced audio is messy in ways CatMeows is not: phone mics, TV noise, other
pets, optimistic labels.

- **VAD / energy gate at intake** rejects clips with no vocalization *before*
  storage, so silence and pocket-dials never enter the set.
- **Store the original, process copies.** Every cleaning step is reversible.
- **Disagreement flagging:** clips where the owner label and a high-confidence
  model prediction disagree are flagged `label_model_disagree_highconf` — these
  are either the most informative samples or mislabels, both worth a human look.
- **Per-owner trust signals:** track each owner's confirm/correct/unknown mix.
  An owner who taps "guessed" on everything in under a second is a noise source
  and can be down-weighted or reviewed.
- **Group by owner, not just cat, in every split.** An owner's phone and room
  are a shared confound across all their cats; holding out whole owners is the
  honest generalization test (this repo already argues the cat-grouping case —
  owner-grouping is the crowdsourced extension of it).
- **Class balance is monitored, not forced.** If `distress` floods in and
  `greeting` is scarce, the dashboard shows it rather than the bot silently
  rejecting common classes.

---

## 7. Security

- Bot token, `server_secret` and storage credentials come from the environment /
  a secret manager. The repo ships a `.env.example` with blank keys and nothing
  real. `.env` is gitignored.
- **`server_secret` rotation** breaks the `telegram_user_id → owner_id` mapping
  by design; document that rotating it orphans existing owners (acceptable, and
  arguably privacy-positive, but must be intentional).
- Webhook endpoint validates Telegram's secret token header; ignore unsigned
  calls.
- Rate-limit per owner id to blunt spam and abuse; cap clip size and duration.
- No audio, account id or message text in logs or error traces. Scrub before
  logging.
- Treat every incoming file as untrusted: validate the container, transcode in a
  sandbox, never execute or trust filenames.

---

## 8. Architecture

Small and boring on purpose.

```
Telegram  ──webhook──▶  Bot service (stateless)
                          │  ├─ consent / command handling
                          │  ├─ VAD gate
                          │  ├─ inference client ─▶  Model service (loads the
                          │  │                        current best probe; §ties to
                          │  │                        src/train_transfer.py)
                          │  ├─ metadata store  ─▶  DB  (owners, cats, clips)
                          │  └─ audio store     ─▶  object storage (originals)
                          ▼
                    Export job (offline, reviewed) ─▶ CC BY 4.0 dataset release
                    Quality dashboard (read-only over the DB)
```

- **Reuse, don't fork, the model path.** The "current best model" is exactly the
  probe from [`src/train_transfer.py`](../src/train_transfer.py) over frozen
  embeddings ([`src/embeddings.py`](../src/embeddings.py)); the model service
  wraps a serialized version. When a better model lands in the repo, the bot
  points at it — one source of truth for "what MeowSense predicts".
- **The bot never trains.** It collects and serves. Training stays in the repo,
  reproducibly, on exported data.
- **Language:** Python, to share `features.py`/`embeddings.py` with the repo. A
  library choice (python-telegram-bot vs aiogram) is deferred — see decisions.

---

## 9. Rollout

Ordered so that the privacy-critical pieces exist before any real audio is
accepted.

- [ ] **M0 — consent + deletion first.** `/start` consent flow, `/delete_my_data`
      that truly deletes, `/privacy`. No collection until this works end to end.
- [ ] **M1 — intake.** Receive audio, VAD gate, store original + de-identified
      metadata. Still no public exposure.
- [ ] **M2 — the loop.** Inference client serving the repo's current best model;
      guessed / wrong / don't-know labels; optional enrichers.
- [ ] **M3 — quality dashboard.** Per-owner agreement, class balance, cat/owner
      counts, disagreement queue.
- [ ] **M4 — first release.** Export to CatMeows-compatible CC BY 4.0 dataset
      **only after ≥100 cats and ≥2000 confirmed clips**, with a written review
      of consent coverage and background-speech risk before publishing.

Success is measured in **cats and owners**, not clips — 2000 clips from 5 cats
would not move the benchmark, and the whole point is breadth.

---

## 10. Open decisions for the maintainer

These are genuinely yours; the design cannot resolve them and a deployment must
not proceed until they are. Flagged because acting on any of them has real-world
consequences.

| # | Decision | Why it's yours | Default if unspecified |
|---|---|---|---|
| 1 | **Jurisdiction & legal review.** Which country's data-protection law governs this, and has a human reviewed the consent text? GDPR (EU users) implies specific rights and possibly a DPO. | I am not a lawyer; publishing others' recordings has real legal weight. | **Blocker.** Do not launch without this. |
| 2 | **Hosting & who operates it.** Where the service and audio live, and who is the accountable data controller. | Ongoing cost, security posture, and legal accountability attach to a real person/entity. | Blocker. |
| 3 | **Bot token & secrets.** Creating the bot (BotFather) and provisioning secrets. | Requires your Telegram account and cannot be done from the repo. | Blocker. |
| 4 | **Retention window.** Keep audio indefinitely (under consent) or auto-expire after N months? | A privacy/utility trade-off only you can set. | Indefinite under standing consent. |
| 5 | **Background human speech.** Publish clips as-recorded (with the warning in §5), or invest in speech suppression before release? | Affects both privacy and dataset realism. | As-recorded, with the §5 warning. |
| 6 | **Moderation.** Who reviews the disagreement/abuse queue, and how often? | Needs a human; unmoderated crowdsourcing degrades. | Undefined — name an owner before M2. |
| 7 | **Budget.** Inference + storage + egress have real cost at scale. | Determines whether M2's live model is feasible or the bot defers guesses. | Undefined. |
| 8 | **Library & serialized model format.** python-telegram-bot vs aiogram; how the probe is frozen for serving. | Low-stakes and reversible, but should be decided before M2. | Maintainer's call at implementation time. |

**Until decisions 1–3 are settled, this stays a document.** Building bot code
before then would produce a thing that looks launch-ready but must not be
launched, which is its own hazard.

---

## Relationship to the rest of the project

- The bot **feeds** the benchmark; it does not replace it. Everything collected
  is trained and evaluated in this repo, under the same grouped-validation
  discipline (extended to group by owner).
- A larger dataset is what unblocks three stalled items at once: validating the
  [RMS gain](ROADMAP.md#4-channel-normalization-done) on data not used to select
  it, firming up the [0.54–0.60 band](../README.md#results), and growing the
  [label space](ROADMAP.md#3-expanding-the-label-space).
- See [CONTRIBUTING.md](../CONTRIBUTING.md). Note the roadmap's warning: this
  workstream needs someone comfortable owning a privacy/consent surface, not
  just bot code.
