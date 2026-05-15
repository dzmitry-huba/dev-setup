---
name: academic-research
description: Create, rewrite, expand, or repair comprehensive academic research whitepapers and technical surveys from papers, official reports, repositories, and implementation evidence. Use when Codex is asked for a self-contained academic report, literature review, systems survey, technique taxonomy, shared notation/preliminaries, source map, evidence matrix, renderer-safe diagrams/math, acronym cleanup, symbol explanations, or iterative research-paper polish across topics such as LLM systems, GPU kernels, distributed training/inference, or other fast-moving technical domains.
---

# Academic Research

## Overview

Use this skill to produce research documents that are self-contained learning guides, not source dumps. The output should explain techniques first, cite evidence close to claims, expose implementation paths for deeper study, and survive Markdown preview in common editors such as VSCode and Codex.

Read `references/quality-bar.md` when the task involves a long whitepaper, diagram/math cleanup, table rendering fixes, parallel research agents, or a final review pass.

## Core Workflow

1. **Freeze scope and date.** State the topic, intended reader, deliverable path, source cutoff date, platform focus, and out-of-scope material. For time-sensitive topics, verify recent facts before drafting.
2. **Preserve seed sources.** If the user provides a first plan, source list, or named examples, carry them forward explicitly. Add new sources without dropping the original seed set unless the user says to remove them.
3. **Build an evidence map before prose.** Group primary papers, official reports/docs/blogs, repositories, and high-signal third-party integrations by concept. Mark unsupported or unpublished frontier-lab details as `inferred` or omit them.
4. **Organize around concepts, not frameworks.** Prefer top-level sections such as techniques, primitive mechanisms, parallelism families, kernels, data layouts, scheduling strategies, or failure modes. Mention frameworks as implementation vehicles inside those sections.
5. **Separate shared preliminaries from category-specific content.** For long surveys, consolidate global evidence policy, notation, tensor/rank symbols, placement states, communication primitives, cost models, terminology rules, and renderer policy into one early top-level section. Do not repeat that setup inside every category; local sections should only add local symbols or contracts.
6. **Decompose into primitives.** For each top-level category, explain the primitive mechanisms first, then provide links to papers/docs/code, and then give implementation examples with enough context to decide where to dive deeper.
7. **Make each section self-contained without duplicating the shared layer.** Add local shapes, formulas, symbol explanations, communication patterns, inputs/outputs, and caveats where they help the reader understand without opening links. Reference the shared preliminaries for globally defined notation and evidence labels.
8. **Use agents only when requested.** If the user explicitly asks for parallel agents, split by disjoint sections or categories. Have each worker write a draft artifact in a temporary folder; integrate centrally in one controlled pass.
9. **Review and polish iteratively.** Expect follow-up passes for correctness, acronyms, formulas, diagrams, table rendering, link labels, source coverage, heading renumbering, and duplicated setup material. Keep each pass scoped and verify mechanically.

## Paper Structure

Use this default structure for a new academic research whitepaper:

1. Abstract and scope
2. How to read the paper and renderer/source conventions
3. Shared preliminaries: evidence methodology, notation, tensor/rank symbols, placement states, collectives, and cost models
4. Conceptual taxonomy or technique map
5. One major section per category
6. Cross-category source/evidence matrix, if it improves readability; otherwise keep source maps local to categories
7. Operational triage guide or decision map
8. Open questions and evidence boundaries
9. Bibliography/source index
10. Verification notes

Use in-place expansions for less-common acronyms at first meaningful use. Add a terminology table only when the user requests one or when a dense glossary is clearly more readable than in-place definitions.

Each major category section should include:

- `Technique` or `Concept`: the central idea in plain language.
- `Primitive Mechanisms`: subsections for smaller reusable techniques.
- `Inputs, Outputs, and Shapes`: tensor/matrix/rank/cache shapes where applicable.
- `Communication`: collectives, peer-to-peer transfers, runtime groups, overlap, and synchronization where applicable.
- `Math / Notation`: formulas only where they clarify contracts; explain every symbol.
- `Explore Further`: primary paper/report/docs links and repository paths with readable link text.
- `Evidence Boundary`: what is directly supported, inferred, or unknown.

Shared preliminaries should include only concepts reused across multiple categories. Good candidates are evidence grades, common notation, mesh/rank coordinates, placement states, collectives, peer-to-peer definitions, first-order communication/memory formulas, renderer rules, and global caveats. Poor candidates are a category's core algorithm, local shape contract, local routing/schedule policy, or family-specific decision guide.

## Evidence Rules

- Prefer primary sources: arXiv papers, official technical reports, model cards, vendor docs/blogs, and official repositories.
- Use third-party integrations only when they are the best public evidence for runtime behavior; label them as such.
- Every row in major source-map tables needs at least one citation and an evidence grade such as `paper`, `official repo`, `official docs/blog/report`, `third-party integration`, or `inferred`.
- Benchmarks must preserve hardware, precision, shapes/context length, baseline, and source scope when available.
- Closed-lab practices should be described only from public evidence. Architecture-implied kernel or systems demands must be labeled as inferred.

## Writing Rules

- Start each section with an explanation before listing frameworks or repos.
- Avoid standalone framework catalogues. If frameworks matter, say what technique they implement and why.
- Use concise, academic prose. Prefer precise caveats over promotional language.
- For less-known acronyms, link the acronym and add a short parenthetical definition at first meaningful use. Do not overload common terms such as LLM, GPU, API, or URL.
- Use reader-friendly link labels for code references, such as "FA4 pipeline helpers" or "DeepGEMM grouped GEMM tests"; do not expose long source paths as the visible link text.
- Keep implementation examples traceable but short. Let links carry the deep code path.

## Math and Diagrams

- Use LaTeX for mathematical formulas. Use `$...$` for inline math when the user's target renderer is VSCode or when they explicitly ask for dollar math; use `\(...\)` only if the target renderer requires it.
- Use `$$...$$` for displayed formulas and LaTeX array/matrix diagrams.
- Explain every symbol after each formula or diagram group with `Symbols:` or `Diagram symbols:`.
- In Markdown tables, use `$...$` for math and avoid literal pipe characters inside math; write `\lvert x\rvert` instead of `|x|`.
- For Mermaid diagrams, do not put raw LaTeX in node labels. Use Unicode/plain notation inside nodes, then add a `Diagram notation key` below the diagram mapping labels to LaTeX.
- Choose Mermaid-only diagrams when the target is VSCode preview or the user asks for editable diagrams. Avoid relying on local SVG/PNG unless the user wants rendered image assets.

## Validation Checklist

Before finalizing, run mechanical checks appropriate to the artifact:

- No unresolved placeholders, conflict markers, empty Markdown links, or trailing whitespace.
- All Markdown tables have consistent pipe counts.
- Math delimiters are balanced and match the target renderer policy.
- No accidental raw LaTeX remains in Mermaid node labels or code spans.
- Each displayed formula/diagram group has a symbol or notation explanation.
- Each major source-map row has a citation and evidence grade.
- Less-known acronyms are expanded at first meaningful use.
- Code-reference link labels are readable rather than full source paths.
- Shared setup material is not duplicated across family sections; category sections start with their specific mechanism and only define local symbols.
- Numbered headings are sequential after structural edits, and any section references still point to the right place.
- Diagrams parse syntactically when a parser is available; avoid long headless render attempts that can hang. Prefer static Mermaid parser checks over browser rendering in constrained environments.
- Final response states what changed, what was verified, and any verification gaps.

## Reusable Patterns

For technique surveys, use this section pattern:

```markdown
## Shared System Model, Notation, and Evidence Policy

Define evidence grades, global symbols, placement states, communication primitives, and cost models used by later sections.

Keep this section generic. Move category-specific algorithms, local shapes, schedules, and decision guides back into their own sections.
```

For category sections, use this pattern:

```markdown
### Primitive Name

Explain the mechanism and why it exists.

Inputs and outputs: describe shapes or state.

Formula:
$$
...
$$

Symbols: define every symbol.

Implementation examples: [Readable code link](https://example.com), [paper](https://example.com).

Evidence boundary: say what is direct, inferred, or unknown.
```

For parallel research agents, use this integration pattern:

- Assign one category per worker and a disjoint draft file path.
- Tell workers they are not alone in the workspace and must not edit the main paper.
- Require each draft to include technique explanation, shapes, formulas/symbols, communication where relevant, sources, and caveats.
- Integrate drafts centrally, preserving existing diagrams, notation policy, source style, and renderer rules.
