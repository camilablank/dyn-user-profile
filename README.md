# This Is *So You*: Do LLMs Form Dynamic Profiles of User Emotions?

## What problem am I trying to solve?
LLMs form detailed profiles of user information that persist across turns in a conversation, currently demonstrated for static traits such as age, gender, and socioeconomic status (Chen et. al., 2024)1. Inspired by this result, I explored whether LLMs also encode dynamic profiles of users’ emotions, i.e. turn-by-turn predictions of the user’s current emotion, and whether these can be manipulated to causally shift the model’s tone and content. My research focused on the following questions: 1) Do the internal states of LLMs contain promptly-updated information about users’ current emotions? 2) If so, can we causally steer LLMs based on these emotions?

## Significance: 
If AI can not only track but manipulate a user’s emotions, misalignment threatens user privacy and well-being.

## Setup:
Using Llama-3.1-8B-Instruct, I performed a classification procedure by training linear logistic probes on the model’s hidden state representations and a causal test by steering representations using control probe weights from each emotion’s classification.
I first created a conversation dataset of 250 multi-turn conversations between a “user” and “AI assistant” using GPT-4o. Each conversation has 10-18 user-assistant turns and 4-6 emotion changes. Each turn is annotated with one of 25 fine-grained emotion labels; however, Cowen and Keltner show that collapsing fine emotion labels into higher-level clusters mitigates label sparsity, reducing the risk of overfitting (2017)2. Thus, I divided the 25 fine labels across the following 6 “buckets”: [positive_high, positive_low, calm_steady, worried, neg_low, neg_high].
For causal testing, I wrote an additional prompts dataset of 18 single-turn user messages on various topics.

## Key Takeaways:
1) Llama-3.1 encodes and updates emotions in its hidden states across turns in a conversation, especially at turns where the ground truth user emotion changes. 2) High-contrast transitions between distinct emotions are more detectable than low-contrast transitions between similar emotions. 3) We can steer the LLM to respond to emotions that are not implied by the user’s prompt by translating its representation along the weight vector of a trained linear logistic control probe.
