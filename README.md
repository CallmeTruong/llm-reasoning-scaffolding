# Monte Carlo Tree Search for LLM Reasoning

A systematic comparison of inference-time reasoning strategies for language models, evaluated on the GSM8K mathematical reasoning benchmark. Methods range from a single-pass baseline to a tree-search algorithm adapted from game-playing AI.

---

## Overview

Standard language models commit to a single output path at inference time, with no mechanism for backtracking or exploring alternative reasoning trajectories. This notebook benchmarks four reasoning strategies — Direct, Chain-of-Thought, Self-Consistency, Best-of-N, and MCTS — and analyzes their trade-offs across accuracy, token consumption, and latency.

The central argument is that **inference-time compute**, not just model parameters, is a meaningful lever for improving reasoning quality. MCTS simulates the behavior of native reasoning models (e.g., OpenAI o1, DeepSeek-R1) by externalizing the search process in Python, without modifying the underlying model weights.

---

## Dataset

**GSM8K** — a benchmark of approximately 8,500 grade school math word problems requiring multi-step arithmetic and logical reasoning.

- Source: `datasets` library (`gsm8k`, `main` split)
- Evaluation subset: 200 problems sampled from the test split
- Answer format: final numeric value extracted via regex from model output

---

## Methods

### 1. Direct Answer (Baseline)
Single-pass generation with no intermediate reasoning. Serves as the lower-bound reference. Accuracy: **35.5%**.

### 2. Chain-of-Thought (CoT)
The model is prompted to produce step-by-step reasoning before stating a final answer. Substantially improves performance over the direct baseline, but is limited to a single reasoning path with no recovery from early errors. Accuracy: **93.0%**.

### 3. Self-Consistency
Runs CoT inference `n_votes` times in parallel; takes the majority answer across runs. More robust to stochastic errors but cannot correct systematic mistakes shared across all samples. Accuracy: **89.5%**.

### 4. Best-of-N
Generates `n` independent CoT candidates, scores each using a verifier prompt (without access to ground truth), and selects the highest-scoring candidate. Susceptible to self-assessment bias, where the model overscores confidently-stated but incorrect solutions. Accuracy: **88.0%**.

### 5. Monte Carlo Tree Search (MCTS)
Constructs a reasoning tree iteratively using the classic MCTS loop. Each node represents a partial reasoning trace; the UCB1 formula balances exploitation of high-value nodes against exploration of less-visited branches.

**MCTS loop:**

| Step | Description |
|------|-------------|
| Selection | Traverse the tree following UCB1 scores to the most promising leaf node |
| Expansion | Generate `k` distinct next reasoning steps via the LLM |
| Rollout | Complete the solution from the current partial state |
| Backpropagation | Propagate the evaluator score up through all ancestor nodes |

**UCB1 formula:**

$$UCB1(node) = \frac{Q(node)}{N(node)} + c \cdot \sqrt{\frac{\ln N(parent)}{N(node)}}$$

where $c = 1.4$ (tunable for language tasks).

Accuracy: **96.5%**.

---

## Results

| Method | Accuracy | Avg Tokens | Avg Latency (s) | Tokens / Correct Answer |
|---|---|---|---|---|
| CoT | 93.0% | 283 | 8.4 | 304 |
| Self-Consistency | 89.5% | 819 | 8.7 | 916 |
| Best-of-N | 88.0% | 1,764 | 9.2 | 2,005 |
| MCTS | 96.5% | 9,156 | 136.4 | 9,488 |

> Raw accuracy is a misleading metric when methods differ substantially in token usage. The **tokens-per-correct-answer** column is the more informative efficiency measure.

---

## Visualizations

The notebook generates the following figures:

- **MCTS search tree** — node size proportional to visit count, color mapped to average reward score, best path highlighted
- **Evaluator score distribution** — histogram and per-question box plots of rollout scores across all MCTS iterations
- **Comparison dashboard** — bar charts for accuracy, average tokens, average latency, and tokens per correct answer
- **Per-question correctness heatmap** — correctness matrix across all methods and questions

---

## Key Observations

**System-level scaffolding vs. native reasoning models.** The methods in this notebook are forms of *system-level scaffolding*: external Python code manages the search process, and reasoning occurs entirely in visible output tokens. This differs from native thinking models (e.g., o1, DeepSeek-R1), which are trained via reinforcement learning to reason in a latent space prior to generating output. MCTS wrapping approximates this behavior but at a higher inference cost.

**The evaluator as a bottleneck.** MCTS quality depends heavily on the reward signal used to score rollouts. This notebook uses the same LLM as a self-evaluator, which introduces self-assessment bias. In production systems, a dedicated Process Reward Model (PRM) would replace this role.

**When MCTS is most beneficial:**
- Problems with misleading initial steps where the first reasoning direction is incorrect
- Tasks requiring multi-step deduction where early errors compound
- Settings with sufficient compute budget, as accuracy scales with `iterations` and `k`

---

## Requirements

```
datasets
litellm
matplotlib
nest_asyncio
numpy
```

Install:
```bash
pip install datasets litellm matplotlib nest_asyncio
```

---

## Setup

Set your API key as an environment variable before running:

```bash
export API_KEY=your_key_here
```

The notebook uses `deepseek/deepseek-chat` via LiteLLM. Any model supported by LiteLLM can be substituted by changing the `model` parameter in `call_llm`.

---

## Usage

Open `MCTS_LLM.ipynb` in Jupyter and run cells sequentially. Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_examples` | 200 | Number of GSM8K problems to evaluate |
| `n_votes` | 3 | Votes for Self-Consistency |
| `n` | 3 | Candidates for Best-of-N |
| `iterations` | 20 | MCTS search iterations per problem |
| `k` | 2 | Child nodes expanded per MCTS iteration |
| `max_depth` | 4 | Maximum tree depth |

Increasing `iterations` and `k` generally improves MCTS accuracy at the cost of proportionally more token usage.

---

## Repository Structure

```
.
├── MCTS_LLM.ipynb          # Main notebook
├── README.md
└── images/
    ├── simulate.png         # MCTS 4-step loop diagram
    └── mcts_tree_q1.png     # Example search tree visualization
```

---

## Notes

- All methods operate under the same total token budget per question to enable fair comparison.
- The MCTS evaluator scores rollout quality on a 0.0–1.0 scale based on logical consistency and arithmetic correctness, without access to the ground truth answer.
- Results are specific to the model, sample, and hyperparameters used. Generalizations should be made with care.
