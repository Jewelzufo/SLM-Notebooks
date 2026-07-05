# SLM-Notebooks
This repository contains educational Colab notebooks that use HuggingFace models for NLP tasks.

# Notebooks

## Prompting Lightweight Granite 4.0 H 350M for JSON Classification

### Overview

A beginners guide to reliable text classification using small local models with structured JSON output. This repository demonstrates prompt engineering techniques, validation strategies, and repair mechanisms specifically designed for 350M parameter models like `unsloth/granite-4.0-h-350m-GGUF`.

>**Project:** [Granite-4.0-H-350M Prompting](https://github.com/Jewelzufo/SLM-Notebooks/tree/main/granite-4.0-h-350m)

---

## Simulated Tool Use Evaluation Granite-4.0-350M

### Summary

This Google Colab notebook evaluates the "unsloth/granite-4.0-h-350m-GGUF" model on simulated tool selection tasks. It uses structured JSON classification, schema validation, automatic output repair, and a labeled benchmark to measure routing accuracy, JSON validity, latency, and inference reliability. Designed for CPU-friendly execution, the notebook provides a reproducible framework for benchmarking lightweight language models as tool-routing components in edge AI and agentic workflows.
