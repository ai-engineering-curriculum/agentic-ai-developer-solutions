# project-103-multi-provider-comparison-report — Solution

Reference walkthrough for the **Multi-Provider Comparison Report** capstone
project. The project asks the student to take the `support-agent` from
mod-105/mod-106 and the golden-set eval harness from mod-109, run it across
two or more LLM providers (Anthropic, OpenAI, and optionally Google), and
produce a reproducible comparison report that drives a model-selection
decision on quality, cost, and latency — not on a vibe.

- **Full walkthrough:** [`SOLUTION.md`](./SOLUTION.md)

The solution composes three earlier building blocks:

| Prerequisite | Role in this project |
|---|---|
| [`mod-106`](../../modules/mod-106-llm-app-deployment/) agent + `AGENT_FAKE=1` stub | System under test |
| [`mod-107 exercise-05`](../../modules/mod-107-streaming-async-and-orchestration/exercise-05-measure-time-to-first-token-and-tail-latency/) percentile harness | Latency measurement discipline |
| [`mod-109 exercise-04`](../../modules/mod-109-minimal-eval-and-regression-checks/exercise-04-regression-report-for-a-model-swap/) regression report | Report structure, policy TOML, exit-code verdicts |

Where mod-109's exercise-04 is a two-way diff between two runs of the *same*
provider, this project generalises to an N-way comparison across *different*
providers, adds cost-per-run as a first-class metric, and treats the
selection decision as the report's output.
