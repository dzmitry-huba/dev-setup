---
name: ascii-design-slides
description: "Create or revise developer-focused technical slide decks with editable text-only slides, JetBrains Mono, dark canvas, large consistent typography, colored emphasis, ASCII architecture diagrams, benchmark ASCII bars, no images, and no speaker notes. Use for Google Slides or PPTX decks when the user asks for text/ASCII-only architecture, design, performance, or systems presentations."
---

# ASCII Design Slides

Use this skill to make developer-facing design presentations with crisp text,
ASCII diagrams, and performance analysis.

## Core Contract

- Use only editable text blocks and native slide text. Do not generate images, raster diagrams, screenshots, SVG art, or speaker notes.
- Use `JetBrains Mono` for every text run.
- Prefer a dark full-slide background and flat canvas; avoid cards, pills, UI chrome, screenshots, decorative blobs, or image assets.
- Use only two slide layouts:
  - single-column, centered content
  - two-column, balanced content
- Keep each slide focused on one aspect. Split overloaded content into more slides, not smaller text.
- Keep content occupying roughly 70-80% of the slide with generous margins.
- Do not split ASCII diagrams across slides or columns.
- Verify every slide visually; fix overlap, clipping, inconsistent sizing, and unreadable text before handoff.

## Visual System

Use a restrained terminal/editor palette:

- Background: near-black navy, e.g. `#0B1020`
- Main text: light slate, e.g. `#D7DEE8`
- Titles: near-white, e.g. `#F8FAFC`
- Cyan emphasis / links / arrows: `#38BDF8`
- Green success / proposed path / improvement: `#34D399`
- Amber warnings / run commands / labels: `#FBBF24`
- Red only for blocked, forbidden, or risk states: `#F87171`
- Muted text: `#94A3B8`

Use bold and color deliberately:

- Bold section labels, slide titles, key claims, and phase numbers.
- Color only the important semantic pieces: proposed path, blocked path, current focus, links, benchmark highlights.
- Keep most body text regular weight.

Recommended font sizes for 16:9 decks:

- Title slide title: 46-52 pt
- Slide titles: 32-38 pt
- Section labels / big claims: 22-26 pt
- Body text: 18-21 pt
- Dense code, runbook, or large ASCII diagrams: 15.5-17 pt minimum
- Keep sizes consistent across the deck; prefer 4-6 total text sizes.

## Narrative Shape

For design decks, use this flow unless the source material demands otherwise:

1. Title: name the system literally; one short subtitle.
2. Motivation: why this exists and what should stay simple.
3. Design goals: numbered constraints and non-goals.
4. Boundary: the key security, data, ownership, or API boundary.
5. Interfaces/configuration: where policy, API shape, or runtime options live.
6. Selection/control plane: how the system chooses the relevant path or mode.
7. State setup: what is established before work begins.
8. Main execution path: producer path, consumer path, signaling, and completion.
9. Resources: allocation, reuse, backpressure, and pressure limitations.
10. Operation examples: one baseline path and one proposed path using the same terminology.
11. Correctness evidence: tests, counters, invariants.
12. Limitations: honest current gaps and rollout blockers.
13. Benchmark setup: hardware, request shapes, methodology.
14. Results: p50 latency first, then throughput, diagnostics, and interpretation.
15. Links/runbook: PR, design doc, benchmark command, output root.

## ASCII Diagram Rules

- Use fixed-width text with `+---+`, `|   |`, `---->`, `v`, and simple labels.
- Keep diagram line lengths within the slide layout:
  - single column: about 70 characters max
  - two column: about 34-40 characters max per column
- Use blank lines to create breathing room in tall diagrams.
- For timelines, place earlier events higher and later events lower.
- Show actors as repeated headings when sequencing matters:

```text
time
  |
  v

Component A / producer
----------------------

Coordinator / runtime
---------------------

Component B / consumer
----------------------

Storage / external dependency
-----------------------------
```

- Use the same terminology across paired slides. For example, do not mix
  `consumer`, `worker B`, and `service B` for the same actor.
- Use consistent indexing notation, such as `state[A]`, `ready[A]`, and
  `buffer[A]`.
- Never place arrows where they visually point from the wrong actor to the wrong
  component. If direction is confusing, redraw with explicit producer and consumer.

## Benchmark Slides

Use p50 latency for the main latency comparison unless the user asks otherwise.
Show absolute numbers and the ratio:

```text
baseline -> proposed p50 latency     proposed / baseline

case A    97ms ->  269ms   2.76x  [#############-------]
```

ASCII bars must match the number:

- Percentage bars: `filled = round(percent / 100 * width)`.
- Ratio bars: choose a max ratio for the slide, disclose or imply it in the axis,
  and use `filled = round(ratio / max_ratio * width)`.
- Keep a constant width per slide, commonly 24 or 26 characters.
- Do not reuse a bar template if the count of `#` no longer matches the value.

Include throughput as retained throughput vs baseline when helpful:

```text
CANDIDATE THROUGHPUT AS % OF BASELINE
higher is better

case A   36.2%  [#########---------------]
```

For diagnostic runs, label exactly what is isolated, e.g.
`feature off / feature on`, `pooled / unpooled`, or `baseline / proposed`; do
not let the slide imply a stronger conclusion than the experiment supports.

## Authoring Workflow

1. Gather the source docs, benchmark numbers, and any existing target deck.
2. Write a slide outline before editing. Each slide should have one job.
3. Draft content in plain text first, including ASCII diagrams.
4. Build slides with editable text boxes only.
5. Apply the visual system consistently: background, font, sizes, colors, and bold.
6. Verify slide-by-slide using rendered thumbnails or screenshots:
   - no overlap or clipping
   - diagrams are intact and not split
   - font sizes are readable and consistent
   - content fills about 70-80% of the slide
   - title placement and margins are consistent
   - no speaker notes were added
7. Iterate until the deck is clean, centered, and readable.

For Google Slides:

- Follow the `google-slides` skill for connector safety, target identity, batch
  update write safety, and thumbnail verification.
- For a new deck, prefer making a local PPTX with the `presentations` skill and
  importing it as native Google Slides when that plugin is available.
- For an existing deck, use Google Slides connector reads/writes directly and
  verify each changed slide with a fresh thumbnail.

For PPTX:

- Follow the `presentations` skill for local deck authoring and render/verify
  loops.
- Keep all diagrams as text, not generated images.

## Quality Bar

The deck is not done until it feels like a coherent design walkthrough:

- motivation and goals come before mechanism
- mechanism comes before benchmarks
- limitations are explicit
- performance claims include setup and absolute numbers
- every diagram is readable at presentation distance
- the final file remains editable text, not flattened slide images
