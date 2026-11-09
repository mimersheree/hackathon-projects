# Topological Blast Walls: Multi-Agent Swarm Containment Benchmark

**Submission for the [Apart Research Swarm Alignment Hackathon 2026](https://apartresearch.com/sprints/swarm-alignment-2026) — Track 2: The Science of Agent Networks**

## Project Overview

As software engineering workflows transition from single-model assistants to autonomous multi-agent swarms, inter-agent communication protocols introduce a critical security risk: **lateral prompt contagion**. When an ingestion agent processes untrusted external artifacts (dependency manifests, scanner logs, or pull request metadata), embedded adversarial instructions can propagate across agent handoffs and compromise downstream execution nodes.

This project designs and builds an end-to-end empirical evaluation harness testing how multi-agent communication topology governs the containment of supply-chain prompt injections in autonomous CI/CD pipelines:

- **Mesh (`mesh`)**: Unrestricted shared blackboard where `Triage`, `Reviewer`, and `Deployer` read and append to a single global state.
- **Linear Chain (`linear`)**: Unidirectional sequential pipeline (`Triage -> Reviewer -> Deployer`) with isolated pairwise handoffs.
- **Gated Hub (`gated_hub`)**: Sequential pipeline integrating an inference-layer policy and schema checkpoint (`Triage -> Reviewer -> Gatekeeper -> Deployer`) that acts as a topological blast wall before execution.

---

## Key Engineering & AI Safety Contributions

- **Multi-Agent Orchestration & Topological Isolation**: Asynchronous evaluation engine (`asyncio`, `google-genai`) implementing configurable communication graphs (`mesh`, `linear`, `gated_hub`) with resumable JSONL trace logging and rate-limit resilience.
- **Inference-Layer Defense & Structured Output Contracts**: Deterministic schema enforcement (`Pydantic` `DeployerDecision`) paired with a `Gatekeeper` verification bottleneck that strips unauthorized shell overrides, external network calls, and secret exfiltration payloads.
- **Capability-Confined Execution Sandbox**: Custom `SandboxExecutor` enforcing workspace path-traversal protection and socket-level network isolation (`socket.socket` interception) to prevent host file access or outbound exfiltration.
- **Adversarial Benchmark Design**: 26-task CI/CD benchmark (`data/tasks.json`) spanning benign engineering workflows, supply-chain metadata overrides (`curl`/`wget`), Python reverse shells, base64-obfuscated `package.json` hooks, and realistic maintainer build-hook perturbations.
- **Statistical & Automated QA Rigor**: Two-proportion $z$-test and Wilson 95% confidence interval pipeline (`scipy.stats`) backed by an 11-test `pytest` verification suite covering negative controls, perturbation schemas, gatekeeper rejection, sandbox containment, and information-flow invariants.

---

## Repository Structure

```text
swarm-containment-eval/
├── data/
│   └── tasks.json               # 26 benchmark tasks (core, perturbation, realistic suites)
├── results/
│   ├── trials.jsonl             # Resumable per-trial Gemini API execution traces
│   ├── eval_results.json        # Aggregated topology metrics and statistical tests
│   └── contagion_results.png    # Publication visualization (Figure 1)
├── tests/
│   └── test_swarm.py            # 11-test pytest suite (negative controls, sandbox, schema, stats)
├── run_eval.py                  # Asynchronous multi-agent topology evaluation harness
├── calc_stats.py                # Two-proportion z-test calculator (scipy.stats)
├── plot_results.py              # Publication-grade chart generator (matplotlib)
├── pytest.ini                   # Pytest async configuration
└── requirements.txt             # Python dependencies
```

---

## How to Run

### 1. Environment Setup

```bash
source .venv/bin/activate
pip install --prefer-binary -r requirements.txt
export GEMINI_API_KEY="your-api-key"
```

### 2. Run the Automated Verification Suite

```bash
pytest -v
```

### 3. Run the Live Evaluation, Statistical Tests & Plot Generator

```bash
python run_eval.py --repeats 2 --suite core
python calc_stats.py
python plot_results.py
```

**Optional Evaluation Flags:**

- Run expanded perturbation or realistic supply-chain suites:
  ```bash
  python run_eval.py --repeats 1 --suite perturbation
  python run_eval.py --repeats 1 --suite realistic
  ```
- Run with command-extractor worker prompts:
  ```bash
  PROMPT_STYLE=extractor python run_eval.py --repeats 1 --suite realistic
  ```
