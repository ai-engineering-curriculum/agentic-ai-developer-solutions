# AI Infrastructure Agentic AI Developer — Solutions Repository

<!-- aicg:site-banner -->
> 🎓 Part of the **free, open-source AI Infrastructure Curriculum**. For live, instructor-led **[cohorts](https://ai-infra-curriculum.github.io/junior.html)** and **[team programs](https://ai-infra-curriculum.github.io/teams.html)**, visit **[ai-infra-curriculum.github.io](https://ai-infra-curriculum.github.io/)**.
<!-- /aicg:site-banner -->

> **Status**: ✅ Reference solutions complete for every module exercise. AI-assisted content under ongoing human review.

## 🎯 Overview

This repository holds the **reference solutions** for the paired
[`ai-infra-agentic-ai-developer-learning`](https://github.com/ai-engineering-curriculum/agentic-ai-developer-learning)
track — the entry rung of the Agentic AI engineering ladder, covering LLM
fundamentals, prompt engineering, tool and function calling, retrieval, your
first reason-act agent, and shipping an LLM-powered application.

Each solution is a **per-exercise reference walkthrough**, not just a code dump.
Every walkthrough follows the same shape:

- **Approach** — how to read the exercise and why the reference is structured the way it is.
- **Reference implementation** — annotated, runnable code (Anthropic Messages API, portable to OpenAI).
- **Meeting the acceptance criteria** — an explicit map from each requirement to the line that satisfies it.
- **Common pitfalls** — the grader-observed mistakes that cost people the exercise.
- **Verification** — copy-paste commands to confirm a green run.

Attempt each exercise yourself first. The value is in **comparing your
structure to the reference** — where the retry policy lives, where chat history
resets, how token counts and costs are measured rather than guessed.

## 📁 Repository Structure

```text
ai-infra-agentic-ai-developer-solutions/
├── modules/
│   └── mod-XXX-*/                       # one directory per learning module
│       ├── README.md                    # module index + exercise links
│       └── <exercise>/README.md         # per-exercise reference walkthrough
├── projects/
│   └── project-XXX-*/README.md          # capstone walkthroughs
├── SOLUTIONS_INDEX.md                   # inventory + completion map
└── README.md                            # this file
```

Each exercise solution lives at
`modules/<mod>/<exercise>/README.md` and mirrors the exercise of the same name
in the learning repository.

## 📚 Modules

Six modules, three exercise solutions each — **18 reference walkthroughs** in
total.

| Module | Solutions |
|---|---|
| [mod-101 — LLM Fundamentals](modules/mod-101-llm-fundamentals/) | 3 |
| [mod-102 — Prompt Engineering & Structured Output](modules/mod-102-prompt-engineering/) | 3 |
| [mod-103 — Tool & Function Calling](modules/mod-103-tool-and-function-calling/) | 3 |
| [mod-104 — Retrieval Basics (Embeddings & Simple RAG)](modules/mod-104-retrieval-basics/) | 3 |
| [mod-105 — Your First Agent: The Reason-Act Loop](modules/mod-105-first-agent/) | 3 |
| [mod-106 — LLM App Deployment](modules/mod-106-llm-app-deployment/) | 3 |
| **Total** | **18** |

Two capstone projects — [project-101 (LLM-powered app)](projects/project-101-llm-powered-app/)
and [project-102 (tool-using agent)](projects/project-102-tool-using-agent/) —
are scaffolded; their full walkthroughs land on upcoming autonomous content
cycles. See [`SOLUTIONS_INDEX.md`](SOLUTIONS_INDEX.md) for the live completion map.

## 📖 How to Use This Repository

### For Self-Study

1. Start in the **learning repository** and work the exercise on your own.
2. Get it running and passing its acceptance criteria before looking here.
3. Open the matching `modules/<mod>/<exercise>/README.md` and read the **Approach** first.
4. Diff your structure against the **Reference implementation** — note where decisions differ.
5. Run the **Verification** commands with a funded API key and a small spend cap.
6. Re-read **Common pitfalls** to confirm you avoided the grader-observed mistakes.

### For Instructors

- Assign exercises from the **learning repository**; keep this repo as the answer key.
- Use the **Approach** and **Meeting the acceptance criteria** sections as lecture and demo scaffolding.
- Lift the **Common pitfalls** lists straight into rubrics and office-hours prep.

### For Hiring Managers

- Use the exercises as **technical-screen baselines** and these walkthroughs as the reference bar.
- Evaluate a candidate's structure against the reference, not just whether the code runs.
- Draw on the pitfalls and acceptance-criteria mappings as **interview discussion material**.

## 🔗 Paired Learning Repository

- **[ai-infra-agentic-ai-developer-learning](https://github.com/ai-engineering-curriculum/agentic-ai-developer-learning)**
  — the learning materials, exercises, and project stubs these solutions answer.
  Work there first; return here to compare.

---

<!-- aicg:maintained-by -->
Maintained by [VeriSwarm.ai](https://veriswarm.ai)
