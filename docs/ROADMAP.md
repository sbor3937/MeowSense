# MeowSense Roadmap

The baselines in this repository establish an honest starting point: on the
CatMeows dataset, with validation grouped strictly by cat, **nothing computed
from the audio itself beats the majority-class baseline (~0.50)** — MFCC
classifiers and a from-scratch CNN all land at 0.49–0.53. The first thing to
clear the bar was a **better prior**: frozen pretrained embeddings + a probe
reach **0.54–0.60** across two independent backbones (workstream 2 below, now
implemented and replicated). Everything here follows from that shape of result —
cheap priors help, more network does not, and since the spread *between*
backbones is smaller than the spread between folds, the remaining ceiling looks
like data and labels rather than architecture.

The three workstreams are ordered by how much they would move the number:

1. **More data** (now the binding constraint) — [Telegram collection bot](#1-telegram-collection-bot)
2. **Better priors** ✅ *(done — AST probe at 0.60)* — [Transfer learning](#2-transfer-learning)
3. **A more useful label space** — [Expanding to 5-6 classes](#3-expanding-the-label-space)

---

## Why the current results look the way they do

Three structural problems, none of which are fixed by tuning hyperparameters:

| Problem | Evidence | Consequence |
|---|---|---|
| **Too few cats** | 21 cats, 440 clips | A held-out fold contains 4-5 cats. One unusual animal swings fold accuracy by 10+ points (observed fold spread: 0.27-0.63). |
| **Context is not fully acoustic** | `food` recall 0.22-0.38 | "Waiting for food" and "brushing" can produce near-identical calls. The label depends on the *situation*, not only the signal. |
| **Individual variation dominates** | Accuracy collapses when moving from random splits to cat-grouped splits | Models learn *who is meowing* faster than *what they mean*. |

The first is a data problem. The second is a label problem. The third is why we
will not relax the grouped-validation rule to make numbers look better.

---

## 1. Telegram collection bot

**Goal:** grow from 21 cats to hundreds, with labels that carry the situational
context the audio alone cannot.

### Why Telegram

Cat owners already record their cats. A bot removes every step between "my cat
just did something" and a labelled sample: no app install, no account, works on
the device already in their hand.

### Interaction loop

```
owner sends voice note / audio
        |
        v
  bot runs current model
        |
        v
  "I think this is: ASKING FOR FOOD (54% confident)"
  [ guessed ]  [ wrong ]  [ don't know ]
        |
        +-- guessed     -> label = prediction        (confirmed)
        +-- wrong       -> "what was it?" -> menu    (corrected)
        +-- don't know  -> stored unlabelled         (semi-supervised pool)
```

The three-button design is deliberate. A plain "what is your cat saying?" prompt
gets abandoned; confirming or correcting a guess takes one tap. **"Don't know"
is a first-class answer** — forcing a choice manufactures noise, and an honest
"don't know" is more useful than a coin-flip label.

### Metadata to capture (optional, one tap each)

- Context menu: food / play / greeting / distress / litter / vet / other
- Time since last meal (bucketed) — disambiguates `food` directly
- Whether a human was interacting with the cat at the time
- Cat ID (per-owner), breed, age, sex, neuter status — **cat ID is mandatory**;
  without it, grouped validation is impossible and the whole dataset inherits
  the leakage problem this project exists to avoid.

### Data quality

Crowdsourced audio is messy in ways CatMeows is not: phone mics, TV noise, other
pets, and owners who label optimistically.

- Store the **original** audio; never overwrite it with a processed version.
- Run VAD-style energy gating to reject clips that contain no vocalization.
- Flag clips where owner label and model prediction disagree *and* model
  confidence is high — these are either the most informative samples or
  mislabels, and both are worth a human look.
- Track per-owner label agreement; an owner who marks everything "guessed"
  without listening is a noise source.
- Hold out **whole owners**, not just whole cats — an owner's recording setup is
  itself a confound.

### Consent, licensing and privacy

Non-negotiable, and the reason this is a design document rather than code:

- Explicit opt-in at first contact, in plain language: what is stored, for how
  long, and that clips may be published in an open dataset.
- A working delete path (`/delete_my_data`) that actually removes audio.
- Voice notes can contain human speech in the background. Either strip it or
  state clearly that clips are published as recorded.
- Release the resulting dataset under CC BY 4.0, matching CatMeows, so the two
  can be pooled without a licence conflict.

### Milestones

- [ ] Bot skeleton: receive audio, store with metadata, consent flow
- [ ] Inference endpoint serving the current best model
- [ ] Feedback loop + label store
- [ ] Quality dashboard (per-owner agreement, class balance, cat count)
- [ ] First public release once ≥100 cats and ≥2000 confirmed clips

---

## 2. Transfer learning

**This is the highest-leverage change available today**, and unlike the bot it
needs no new data.

`src/train_cnn.py` documents the ceiling of learning from scratch: 23k
parameters on ~300 clips only manage to *tie* a one-line MFCC SVM (~0.53), at a
large compute premium and still within noise of the baseline. That is the signal
to stop scaling the network. Models pretrained on AudioSet (~2M clips) already
encode "what animal vocalizations sound like"; we only need to learn the last
step.

| Model | Pretraining | Notes |
|---|---|---|
| **YAMNet** | AudioSet | MobileNet-based, CPU-friendly, has native `Cat`/`Meow` classes. Cheapest thing to try — start here. |
| **PANNs** (CNN14) | AudioSet | Consistently strong transfer on bioacoustics; heavier than YAMNet. |
| **AST** | AudioSet | Transformer, usually the best of the three, but the most data-hungry to fine-tune and the easiest to overfit on 440 clips. |

### Approach

1. **Frozen embeddings first.** Extract embeddings, train logistic regression /
   SVM on top. Fast, hard to overfit, and an honest ceiling check on what the
   pretrained representation already knows.
2. **Then partial fine-tuning.** Unfreeze the last block or two with a low
   learning rate.
3. Only then consider full fine-tuning, with the strong expectation that it
   overfits.

### Resampling caveat

All three models expect 16 kHz; CatMeows is 8 kHz. Upsampling does not restore
the missing 4-8 kHz band, which is exactly where much feline vocalization energy
sits. Expect a real penalty and **do not compare these numbers against
literature trained on native 16 kHz audio**. This is also an argument for
recording the crowdsourced dataset at 16 kHz or higher from day one.

### Success criteria

Frozen embeddings + linear probe should clear **0.60 on unseen cats** (vs ~0.50
today) to justify the added dependency. If it does not, the ceiling is the
labels, not the model — which would itself be a finding worth publishing.

### Result (implemented — v0.2.0, replicated in v0.3.0)

**The frozen probe cleared the bar: AST embeddings + SVM reach 0.60 ± 0.10 on
unseen cats** (`src/embeddings.py`, `src/train_transfer.py`), the first model in
this repository to beat the majority-class baseline.

**It was then checked against an independent backbone**, because one model
scoring well is a result about that model. CLAP (HTSAT encoder, audio-text
contrastive, 48 kHz) differs from AST in architecture, objective and input rate:

| Backbone | Probe | Accuracy | vs baseline | isolation F1 | food F1 |
|---|---|---|---|---|---|
| AST | SVM-RBF | **0.60 ± 0.10** | +0.10 | 0.76 | 0.35 |
| AST | LogReg | 0.58 ± 0.08 | +0.08 | 0.73 | 0.37 |
| CLAP | LogReg | 0.56 ± 0.07 | +0.06 | 0.71 | 0.34 |
| CLAP | SVM-RBF | 0.54 ± 0.08 | +0.04 | 0.71 | 0.28 |

What this says:

- **The direction replicates.** All four configurations beat the 0.50 baseline
  and every from-scratch model. Pretrained priors genuinely help here.
- **The magnitude does not.** CLAP tops out at 0.56; AST wins only 3 of 5 folds
  head-to-head (mean delta +0.04). Even which probe wins flips by backbone.
  Report the 0.54–0.60 band, not 0.60 alone.
- **Backbone spread (0.04) < fold spread (0.07–0.10).** With 21 cats, the
  dataset limits the number more than the model does — which is why workstream 1
  (more data) is now the priority over a better backbone.
- `food` fails on **both** backbones (F1 ≤ 0.37). Two unrelated models trained on
  millions of clips both missing it is the strongest evidence yet that food is a
  label/context problem, not a representation problem.
- All of this holds **despite** the resampling handicap below.

### Milestones

- [x] `src/embeddings.py` — frozen embedding extraction with on-disk caching
- [x] Linear probe on frozen embeddings, same GroupKFold protocol
- [x] Results in the README table under the *same* validation protocol
- [x] Second, independent backbone (CLAP) as a replication check
- [ ] YAMNet / PANNs — both need a TensorFlow (or extra weights) path; the two
      PyTorch-native backbones already agree, so this is now low priority
- [ ] Partial fine-tuning experiment (unfreeze the last block or two). Note the
      cost: an AST forward pass alone is ~4 s/clip on CPU, so backprop over
      5 folds is impractical without a GPU.
- [ ] Probe the pre-projection CLAP encoder state (768-d) instead of the
      projected 512-d embedding — chosen up front here to avoid selecting a
      variant on test performance

---

## 3. Expanding the label space

CatMeows' three contexts are what its authors could stage in a controlled
setting. They are not the three things owners most want to distinguish, and
`food` vs `brushing` confusion suggests the current partition does not carve
the space at its joints.

Target label set (pending enough crowdsourced data to support it):

| Class | Notes |
|---|---|
| `food` | Present in CatMeows |
| `isolation / distress` | Present in CatMeows; the most acoustically distinct |
| `brushing / handling` | Present in CatMeows |
| `greeting` | Short chirps/trills; acoustically distinct, owners report it reliably |
| `play / hunting` | Chattering, chirping at prey |
| `pain / illness` | **High value, hardest to collect.** Needs veterinary confirmation, not owner opinion. Treat as a separate supervised effort. |

### Constraints

- Do not add a class until there are ≥200 clips from ≥20 distinct cats for it.
  Adding an underpopulated class makes every reported number worse and less
  interpretable.
- Expect `greeting` to be easy and `pain` to be both rare and ethically loaded.
  A model that under-detects pain is dangerous if presented as a health tool;
  scope it as research, and say so loudly in any user-facing surface.
- Revisit whether these should be mutually exclusive at all. Multi-label may fit
  reality better — a cat can greet *and* ask for food in one breath.

---

## Non-goals

Stated explicitly, because open bioacoustics projects tend to drift here:

- **A "cat translator" product.** The honest result is ~0.50 on unseen cats.
  Marketing that as translation would be dishonest, and it is precisely what the
  closed apps in this space already do.
- **Beating published CatMeows numbers by relaxing validation.** Cat-grouped
  splits stay. A higher number obtained by dropping them is not a better model.
- **Real-time on-device inference.** Interesting, but pointless before accuracy
  justifies deployment.

---

## Contributing to the roadmap

Item 2 (transfer learning) is self-contained, needs only the existing dataset,
and is the best entry point for a new contributor. Item 1 needs someone
comfortable owning a privacy/consent surface, not just bot code.

See [CONTRIBUTING.md](../CONTRIBUTING.md).
