# Dynamic User Modeling in Large Language Models

## Overview
This mechanistic interpretability project investigates whether **Large Language Models (LLMs) form dynamic models of users**. That is, whether they internally track user attributes such as knowledge, emotions, trust, and confidence that change across multiple conversation turns.

The goal is to:
1. **Detect**: Identify representations of user attributes in model activations.
2. **Track**: Determine whether these representations update when the user’s state changes mid-conversation.
3. **Intervene**: Apply causal editing to alter the model’s perception of the user and measure the downstream behavioral effect.

This work extends *Chen et al. (2024)* by focusing on **transient, dynamic attributes** rather than static persona traits.

---

## Background & Motivation
LLMs often adapt to the user’s behavior in subtle ways.  
Recent work has shown that:
- They **infer user demographics and intent** from minimal input.
- They **track belief states** across multi-turn dialogues.
- They can be **probed and causally edited** to change these internal representations.

---

## Planned Experiments

### **Phase A — Detection**
- Use **linear probes** on hidden states to identify neurons/directions correlated with a target user attribute.
- Attributes tested:
  - Knowledge state (`knows_fact` vs. `does_not_know_fact`)
  - Emotional state (`happy` vs. `neutral` vs. `sad`)
  - Self-perception (`confident` vs. `neutral` vs. `insecure`)
  - Trust (`trusting` vs. `neutral` vs. `skeptical`)
- Train probes on one-turn conversations; test on later turns to see if attribute is tracked.

### **Phase B — Dynamics**
- Introduce mid-dialogue attribute changes (e.g., user learns a fact or changes emotion).
- Track probe outputs turn-by-turn to measure whether internal representation updates.

### **Phase C — Causal Editing**
- Use **ROME** (Meng et al., 2022) or activation patching to change the model’s representation of the user’s state.
- Measure downstream effects on:
  - Generated responses
  - Chain-of-thought reasoning
  - Persona consistency
