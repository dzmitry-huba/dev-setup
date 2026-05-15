---
name: write-academic-whitepaper
description: Create or rewrite detailed academic-style technical white papers about a software project, system, codebase, architecture, research prototype, or implementation. Use when Codex is asked to produce a self-contained paper, architecture report, technical research note, implementation-grounded explanation, or to rewrite an existing paper with academic structure, rigorous prose, diagrams, terminology tables, implementation citations, source links relative to the generated paper, and GitHub code links.
---

# Write Academic Whitepaper

## Overview

Use this skill to turn a project, codebase, or existing draft into a self-contained academic-style white paper. Optimize for a reader who should understand the system without opening source code, while still being able to trace major claims to implementation references.

## Core Workflow

1. Establish scope and audience.
   - Identify the project or subsystem, the intended reader, and whether the output is a new paper or a rewrite.
   - State any out-of-scope topics explicitly, especially training-only mechanisms, legacy compatibility paths, deprecated modules, or speculative behavior.
   - If the user asks for repo-grounding, inspect implementation files before writing claims.

2. Build an evidence map before drafting.
   - Search for configuration entry points, runtime orchestration, core data structures, major operators, tests, and docs.
   - Group evidence by conceptual topic, not by file order.
   - Distinguish what is directly known from code from what is inferred from naming, call structure, tests, or common systems design.

3. Draft the paper as self-contained prose.
   - Explain concepts in the main sections without requiring the reader to inspect code.
   - Introduce terms before using them heavily.
   - Use examples and simple arithmetic when layout, sharding, scaling, capacity, or rank mapping are central.
   - Prefer precise, neutral claims over promotional language.

4. Add implementation notes inside each major section.
   - End each major section with a next-level subsection named like `### N.M Implementation Notes`.
   - Keep implementation notes concise and evidence-oriented.
   - Link to actual source files with paths relative to the generated paper's location in the repo, and add GitHub links when a GitHub remote can be determined.
   - Do not make a single appendix of code pointers unless the user asks for an appendix.

5. Add inline academic-style citations from main prose to implementation notes.
   - Use clickable citations in the main text, for example `[[12]](#impl-12)`.
   - Add matching implementation-note entries, for example:

```markdown
- <a id="impl-12"></a>[12] Sampling configuration strings are assembled by
  `InstantiatedSampleModelConfiguration.config` in
  [types.py](../lib/project/types.py)
  ([GitHub](https://github.com/<owner>/<repo>/blob/<remote-visible-revision>/lib/project/types.py)).
```

   - Place citations next to the specific claim or mechanism they support, not only at paragraph ends.
   - Use one citation number per distinct implementation anchor. Reuse a citation when the same source supports repeated related claims.
   - Ensure every `[[n]](#impl-n)` has exactly one matching `<a id="impl-n"></a>[n]` entry.

## Recommended Structure

Adapt this outline to the project. Keep section names descriptive and academic:

1. Abstract
2. Introduction
3. Problem Setting or System Model
4. Terminology
5. Architecture or Configuration Flow
6. Core Subsystems
7. Execution Lifecycle
8. Efficiency, Correctness, Reliability, or Evaluation
9. Limitations, Known Ambiguities, or Code-vs-Inference Notes
10. Conclusion

For implementation-heavy papers, place `Implementation Notes` subsections after the main prose of each major section. For example:

```markdown
## 5. Runtime Execution

Main academic prose with inline implementation citations such as [[18]](#impl-18).

### 5.1 Implementation Notes

- <a id="impl-18"></a>[18] Runtime scheduling is implemented in
  [scheduler.py](../src/runtime/scheduler.py)
  ([GitHub](https://github.com/<owner>/<repo>/blob/<remote-visible-revision>/src/runtime/scheduler.py)).
```

## Writing Style

- Write as a research paper or systems white paper: precise, layered, and explanatory.
- Keep the main body self-contained. Code references should support claims, not carry the explanation.
- Define project-specific terminology in a table when there are many local names.
- Use diagrams when they clarify flow, topology, lifecycle, or data movement. Mermaid is usually appropriate for Markdown.
- Use mathematical notation sparingly and explain every symbol.
- Use "known from implementation" and "inferred" language when evidence is incomplete.
- Avoid marketing prose, unexplained acronyms, and unsupported superlatives.
- Avoid long code excerpts. Cite files and symbols instead.

## Implementation Reference Standards

- Cite the narrowest stable implementation anchor: a file plus a class, function, constant, config object, test, or module path.
- Prefer actual source files over directories. Use directories only when a subsystem is represented by many files and no single entry point is honest.
- Use local links that are relative to the generated paper's directory, not absolute workstation paths. Do not hardcode `/Users/...`, `/home/...`, workspace-specific roots, or other machine-local prefixes.
- Determine the repo root, the paper path, and each cited source path. Express source links relative from the paper's directory to the source file inside the same local repo.
- Add a GitHub link beside the relative local link whenever the repo has a GitHub remote. Prefer a commit permalink for stable citations, but only use a commit that is visible to the GitHub remote. Do not use a local-only `HEAD` commit unless you have verified that GitHub can resolve it.
- Derive GitHub URLs from the actual repository remote and a remote-visible revision. Convert SSH or HTTPS remotes to browser URLs, preserve GitHub Enterprise hosts when present, and append `/blob/<remote-visible-revision>/<repo-relative-source-path>`.
- Select the GitHub revision carefully:
  - First identify the local source revision you would like to cite.
  - Verify it is remote-visible with a command such as `git branch -r --contains <commit>`, `git ls-remote <remote> <commit>`, the GitHub API, or an authenticated `gh` check.
  - If the desired commit is local-only or not yet pushed, do not use it in GitHub URLs because those links will 404. Instead, use a remote-visible ancestor, such as the merge base with the default branch, if every cited source path and line anchor exists there.
  - If the source file only exists in local-only changes and no remote-visible revision contains it, omit the GitHub link for that citation and note the omission in verification.
  - If the user explicitly wants branch-relative links, use a remote branch name only after confirming the branch exists on the remote and contains the cited paths.
- Do not leave placeholder values such as `<owner>`, `<repo>`, `<commit>`, `<remote-visible-revision>`, or example paths in the generated paper.
- If no GitHub remote is available, include only the relative local repo link and say in verification that GitHub links could not be generated.
- Format implementation-note file links like this:

```markdown
[layout.py](../pkg/layout.py)
([GitHub](https://github.com/<owner>/<repo>/blob/<remote-visible-revision>/pkg/layout.py))
```

- If citing a symbol, put the symbol in backticks near the link.
- If line numbers were verified and useful, use line anchors in the GitHub URL and keep the local link file-oriented unless the target renderer supports local line suffixes:

```markdown
[layout.py](../pkg/layout.py)
([GitHub](https://github.com/<owner>/<repo>/blob/<remote-visible-revision>/pkg/layout.py#L42))
```

- Keep implementation-note entries short. They should summarize what the citation proves and where to inspect it.

## Rewrite Procedure

When rewriting an existing paper:

1. Preserve accurate content, but reorganize around the reader's conceptual path.
2. Move source lists, evidence dumps, and code-pointer appendices into relevant section-level implementation notes unless the user requests a bibliography-style appendix.
3. Convert broad references into granular inline citations tied to specific implementation-note entries.
4. Remove claims that cannot be supported or label them as inferred.
5. Make the paper readable from top to bottom without chasing links.

## Verification Checklist

Before finalizing:

- Confirm the document has a clear academic structure and self-contained main prose.
- Confirm each major section with code-grounded claims has an `Implementation Notes` subsection.
- Confirm every inline citation has a matching implementation-note anchor.
- Confirm every implementation-note anchor is cited at least once.
- Confirm relative local file links resolve from the generated paper's directory.
- Confirm GitHub links are generated from the repo remote and a remote-visible commit or branch, or explicitly note why they were omitted.
- Confirm GitHub links do not 404: verify the chosen revision exists on GitHub, each linked repo-relative path exists at that revision, and every `#L...` line anchor is in range. Prefer an authenticated `gh api` or equivalent check for at least the revision and representative paths, plus a local `git cat-file`/`git show` check for all linked paths at the same revision.
- Confirm diagrams render syntactically when possible.
- Confirm uncertain claims are labeled as inferred or removed.
- Confirm the final response summarizes what changed and notes any verification gaps.
