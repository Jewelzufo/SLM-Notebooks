# SLM-Notebooks

This repository contains educational Google Colab notebooks demonstrating how to leverage lightweight Hugging Face models for Natural Language Processing (NLP) tasks.

---

## 📌 Table of Contents
* [1. Prompting Lightweight Granite 4.0 H 350M for JSON Classification](#1-prompting-lightweight-granite-40-h-350m-for-json-classification)
  * [Overview](#overview)
* [2. Simulated Tool Use Evaluation (Granite-4.0-350M)](#2-simulated-tool-use-evaluation-granite-40-350m)
  * [Summary](#summary)

---

## 📚 Notebooks

### 1. Prompting Lightweight Granite 4.0 H 350M for JSON Classification

#### Overview
A beginner-friendly guide to implementing reliable text classification using small, local models optimized for structured JSON output. This notebook demonstrates practical prompt engineering techniques, robust validation strategies, and automatic repair mechanisms tailored specifically for 350M parameter models.

* **Target Model:** `unsloth/granite-4.0-h-350m-GGUF`
* **Key Concepts:** Prompt engineering, schema validation, output repair, structured JSON.

> **Project Link:** [Granite-4.0-H-350M Prompting](https://github.com/Jewelzufo/SLM-Notebooks/tree/main/granite-4.0-h-350m)

---

### 2. Simulated Tool Use Evaluation (Granite-4.0-350M)

#### Summary
This notebook evaluates the performance of the 350M Granite model on simulated tool selection and routing tasks. It utilizes structured JSON classification, schema validation, automatic output repair, and a labeled benchmark to systematically measure routing accuracy, JSON validity, latency, and inference reliability. 

Designed entirely for CPU-friendly execution, this notebook provides a reproducible framework for benchmarking lightweight language models acting as tool-routing components in edge AI and agentic workflows.

* **Target Model:** `unsloth/granite-4.0-h-350m-GGUF`
* **Key Concepts:** Tool routing, agentic workflows, benchmarking, CPU-optimized inference.

> **Notebook Link:** [Granite-4.0-H-350m Tool Eval](https://github.com/Jewelzufo/SLM-Notebooks/blob/main/granite4-h-350m-tool-eval/granite_350m_simulated_tool_use_eval_colab.ipynb)
