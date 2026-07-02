# Prompting Lightweight Granite 4.0 H 350M for JSON Classification

A beginners guide to reliable text classification using small local models with structured JSON output. This repository demonstrates prompt engineering techniques, validation strategies, and repair mechanisms specifically designed for 350M parameter models like `unsloth/granite-4.0-h-350m-GGUF`.

## Overview

This project teaches a practical, repeatable pattern for basic text classification that returns structured JSON. The focus is on small, reliable classification tasks where:

- Models are lightweight (350M parameters)
- Output must be valid JSON
- Prompts are narrow and explicit
- Validation and recovery mechanisms prevent common failures

**Key improvement over naive approaches**: Fixes the `ValueError: No JSON object found` error by implementing:
1. Strong few-shot prompts ending with `JSON:`
2. Raw-output debugging
3. Brace-aware JSON extraction
4. Schema validation
5. Safe bare-label coercion
6. Retry-and-repair logic
7. Tutorial fallback for continuity

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Core Concepts](#core-concepts)
- [Usage Examples](#usage-examples)
  - [Sentiment Classification](#sentiment-classification)
  - [Ticket Routing](#ticket-routing)
  - [Email Triage](#email-triage)
- [Adapting for Your Task](#adapting-for-your-task)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)
- [Best Practices Checklist](#best-practices-checklist)
- [License](#license)

## Setup

### Environment Requirements
- Google Colab (CPU runtime recommended for portability)
- Python 3.8+
- Minimal dependencies:
  - `huggingface_hub`
  - `jsonschema`
  - `pandas`
  - `tqdm`
  - `llama-cpp-python` (CPU wheel)

### Installation
```bash
# Install core dependencies
pip install --upgrade huggingface_hub jsonschema pandas tqdm

# Install llama-cpp-python with CPU optimization
pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Model Selection
We use the GGUF-quantized model: `granite-4.0-h-350m-UD-Q4_K_XL.gguf` from `unsloth/granite-4.0-h-350m-GGUF`

This choice balances:
- Colab CPU compatibility
- Quality (better than aggressive 1/2-bit quantizations)
- Reasonable size (~200MB)

To verify available models:
```python
from huggingface_hub import list_repo_files
print([f for f in list_repo_files("unsloth/granite-4.0-h-350m-GGUF") if f.endswith(".gguf")])
```

## Core Concepts

### Why Small Models Need Special Prompting
350M parameter models struggle with:
- Long reasoning chains
- Ambiguous label definitions
- Unstructured output requests
- Overly complex schemas

**Practical Rule**: Small models excel when:
- Task is narrow (≤5 labels)
- Labels are explicit and mutually exclusive
- Output schema is minimal (1-3 flat fields)
- Prompts avoid unnecessary reasoning

### The JSON-First Approach
Instead of hoping for JSON, we engineer prompts to *demand* it:
1. End prompts with `JSON:` to create a clear continuation point
2. Provide exact output examples matching the schema
3. Use deterministic generation (`temperature=0.0`)
4. Validate outputs before trusting them
5. Repair failures with targeted prompts

## Usage Examples

### Sentiment Classification
Classifies text as `positive`, `negative`, or `neutral`.

```python
from llama_cpp import Llama
import json
from jsonschema import validate, ValidationError

# Model initialization (run once)
llm = Llama(
    model_path="path/to/granite-4.0-h-350m-UD-Q4_K_XL.gguf",
    n_ctx=1024,
    n_threads=2,
    n_gpu_layers=0,
    verbose=False
)

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]}
    },
    "required": ["label"],
    "additionalProperties": False
}

def classify_sentiment(text: str) -> dict:
    """Classify sentiment with validation and repair."""
    prompt = f"""
You are a strict sentiment classification function.
Return only valid JSON. No markdown. No explanation.

Allowed labels:
- positive: praise, satisfaction, gratitude, approval, or success
- negative: complaint, frustration, rejection, failure, or dissatisfaction
- neutral: factual, unclear, mixed, or no strong sentiment

Schema:
{{"label":"positive|negative|neutral"}}

Examples:
Input: "The setup was simple and worked perfectly."
JSON: {{"label":"positive"}}

Input: "The login flow failed twice and wasted my time."
JSON: {{"label":"negative"}}

Input: "The ticket was opened on Tuesday."
JSON: {{"label":"neutral"}}

Input: {text!r}
JSON:
""".strip()
    
    # Generation with repair logic (full implementation in notebook)
    # Returns validated JSON dict or raises informative error
    ...
```

**Try it**:
```python
print(classify_sentiment("The setup was simple and the output worked perfectly."))
# Output: {'label': 'positive'}
```

### Ticket Routing
Routes support tickets to categories with urgency flags.

```python
TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["billing", "technical_support", "account_access", "sales", "other"]
        },
        "urgent": {"type": "boolean"}
    },
    "required": ["category", "urgent"],
    "additionalProperties": False
}

def classify_ticket(text: str) -> dict:
    """Route ticket with category and urgency prediction."""
    prompt = f"""
You are a strict support ticket routing function.
Return only valid JSON. No markdown. No explanation.

Choose exactly one category:
- billing: invoices, payment, refund, charge, subscription price
- technical_support: bug, error, crash, broken feature, setup problem
- account_access: login, password, locked account, MFA, permissions
- sales: pricing question before buying, demo request, plan comparison
- other: anything else

Urgent is true only when the user says production is blocked, service is down, security is at risk, or a deadline is immediate.

Schema:
{{"category":"billing|technical_support|account_access|sales|other","urgent":true|false}}

Examples:
Ticket: "I was charged twice."
JSON: {{"category":"billing","urgent":false}}

Ticket: "The API is down in production."
JSON: {{"category":"technical_support","urgent":true}}

Ticket: {text!r}
JSON:
""".strip()
    # ... (repair logic)
    ...
```

**Example**:
```python
print(classify_ticket("We cannot log in after enabling MFA. Production deploy is blocked."))
# Output: {'category': 'account_access', 'urgent': True}
```

### Email Triage
Classifies emails into action requirements.

```python
EMAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["reply_needed", "archive", "escalate"]}
    },
    "required": ["action"],
    "additionalProperties": False
}

def classify_email(text: str) -> dict:
    """Determine required email action."""
    prompt = f"""
You are a strict email triage classifier.
Return only valid JSON. No markdown. No explanation.

Allowed actions:
- reply_needed: sender asks a question or expects a response
- archive: informational message, receipt, newsletter, or no action needed
- escalate: legal, security, angry customer, executive request, or urgent business risk

Schema:
{{"action":"reply_needed|archive|escalate"}}

Examples:
Email: "Can you send the updated invoice before Friday?"
JSON: {{"action":"reply_needed"}}

Email: "Your package was delivered at 2:14 PM."
JSON: {{"action":"archive"}}

Email: "The customer is threatening to cancel unless this outage is resolved today."
JSON: {{"action":"escalate"}}

Email: {text!r}
JSON:
""".strip()
    # ... (repair logic)
    ...
```

## Adapting for Your Task

Follow this template to create new classifiers:

1. **Define a minimal schema** (1-3 flat fields)
   ```python
   YOUR_SCHEMA = {
       "type": "object",
       "properties": {
           "field_name": {"type": "string", "enum": ["option1", "option2"]}
       },
       "required": ["field_name"],
       "additionalProperties": False
   }
   ```

2. **Build a prompt ending with `JSON:`**
   ```python
   def build_your_prompt(text: str) -> str:
       return f"""
   You are a strict [task description] function.
   Return only valid JSON. No markdown. No explanation.

   Allowed [field_name]:
   - option1: [clear definition]
   - option2: [clear definition]

   Schema:
   {{"field_name":"option1|option2"}}

   Examples:
   Input: "<example for option1>"
   JSON: {{"field_name":"option1"}}

   Input: "<example for option2>"
   JSON: {{"field_name":"option2"}}

   Input: {text!r}
   JSON:
   """.strip()
   ```

3. **Use the repair classifier**
   ```python
   def classify_your_text(text: str, debug: bool = False) -> dict:
       return classify_with_repair(
           text=text,
           prompt_builder=build_your_prompt,
           schema=YOUR_SCHEMA,
           max_tokens=60,  # Adjust based on output complexity
           single_enum_field="field_name",  # Only for single-field schemas
           retries=2,
           debug=debug
       )
   ```

4. **Test with diverse examples**
   ```python
   test_cases = [
       "Clear example for option 1",
       "Ambiguous case that tests boundaries",
       "Edge case with unusual phrasing"
   ]
   for case in test_cases:
       print(f"Input: {case}")
       print(f"Output: {classify_your_text(case)}")
   ```

## Debugging and Troubleshooting

### Common Issues and Fixes
| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ValueError: No JSON object found` | Model returned empty text/bare label | 1. Verify prompt ends with `JSON:`<br>2. Enable `debug=True` to see raw output<br>3. Check if labels in schema match model output |
| Incorrect labels despite valid JSON | Label definitions too similar | 1. Make label definitions more distinct<br>2. Add more contrasting examples<br>3. Consider merging overlapping labels |
| Frequent repair attempts | Prompt too vague or complex | 1. Shorten prompt to essentials<br>2. Reduce label count<br>3. Increase example specificity |
| Slow generation | Excessive `max_tokens` | Set `max_tokens` to 2-3x expected output length |

### Debugging Workflow
1. Run classification with `debug=True`:
   ```python
   result = classify_your_text("your text", debug=True)
   ```
2. Examine the **RAW MODEL OUTPUT** section
3. Identify failure pattern:
   - Empty output → Strengthen prompt continuation
   - Bare label → Verify `single_enum_field` usage
   - Extra text → Tighten examples and schema hint
   - Wrong fields → Check schema vs. prompt examples
4. Adjust prompt/schema and retest

## Best Practices Checklist

Before blaming model capacity, verify your task design:

- [ ] **Label count ≤ 5** (ideal: 2-4)
- [ ] **Labels are mutually exclusive** (no semantic overlap)
- [ ] **Schema is one flat JSON object** (no nesting)
- [ ] **Generation uses `temperature=0.0`**
- [ ] **`max_tokens` is minimal** (2-5x expected output)
- [ ] **Every label has a clear, distinct definition**
- [ ] **Prompt ends with `JSON:`**
- [ ] **Examples show exact output shape**
- [ ] **Output validated with `jsonschema`**
- [ ] **Raw failures inspected with `debug=True`**
- [ ] **Tested on labeled examples for accuracy**

For 350M parameter models, the optimal prompt is:
- Short (< 200 tokens)
- Explicit (no room for interpretation)
- Boring (avoids creative reasoning)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Model: [unsloth/granite-4.0-h-350m-GGUF](https://huggingface.co/unsloth/granite-4.0-h-350m-GGUF)
- Framework: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Validation: [jsonschema](https://github.com/Julian/jsonschema)

---

*This README provides a comprehensive overview. For executable code and detailed implementations, refer to the original Colab notebook: `granite_350m_json_classification_colab_v2.ipynb`.*

[Notebook](https://github.com/Jewelzufo/SLM-Notebooks/blob/main/granite-4.0-h-350m/granite_350m_json_classification_colab_v2.ipynb)
