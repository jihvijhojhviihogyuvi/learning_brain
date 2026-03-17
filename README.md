# 🧠 Learning Brain

> A self-learning neural network that grows new neurons! Like GGUF, but dynamic.

**Learning Brain** is a Python package that provides neurogenesis and continuous learning for AI models. It's like GGUF, but instead of being static, it grows new neurons when it learns!

This package includes the **SELSC Engine v3** - with AI-modulated dopamine!

## ✨ Features

- 🧬 **Neurogenesis** - Automatically grows new neurons when confused
- 🧠 **STDP Learning** - Spike-Timing-Dependent Plasticity for realistic neural learning
- 🤖 **AI Dopamine** - Uses pretrained reward model to decide dopamine levels
- 📝 **Sentiment Detection** - Detects praise/correction from user
- 🔗 **Hebbian Layer** - Quick session associations ("cells that fire together, wire together")
- 💾 **Persistent State** - Save/load learned brain states to `.brain` or `.neuro` files
- 🔄 **Model Agnostic** - Works with any HuggingFace model (Qwen, Llama, Phi, etc.)
- 📦 **pip Installable** - Easy to install and use

## 🚀 Installation

```bash
pip install learning-brain
```

## 📖 Usage

### Python API - Simple Brain

```python
from learning_brain import Brain

# Create a new learning brain
brain = Brain("my-model")

# Teach it something
result = brain.learn("The sky is blue and the grass is green")
print(result)
# {'tokens': 9, 'active_neurons': 100, 'neuron_grown': False, 'interactions': 1}

# Save the learned brain
brain.save()

# Check brain info
print(brain.info)
# {'model': 'my-model', 'path': 'my-model.brain', 'active_neurons': 100, ...}

# Later, load the brain
brain = Brain.load("my-model.brain")
```

### Python API - EvolvedChat (with LLM)

```python
from learning_brain import EvolvedChat

# Create chat with any HuggingFace model
# Full HuggingFace path required:
chat = EvolvedChat(model="Qwen/Qwen3-0.6B")

# Chat - brain learns automatically
response = chat.chat("Hello! Teach me about physics.")
print(response)

# Save brain (and optionally model weights)
chat.save("my_ai.neuro")

# Load later
chat = EvolvedChat.load_neuro("my_ai.neuro")
```

Supported model formats:
- `"Qwen/Qwen3-0.6B"`
- `"meta-llama/Llama-3.2-1B"`
- `"microsoft/Phi-3-mini-4k-instruct"`
- Any HuggingFace causal LM

### CLI

```bash
# Run interactive session
python -m learning_brain run Qwen/Qwen3-0.6B

# Create a new brain
python -m learning_brain create mymodel
```

## 🧬 How It Works

### Neurogenesis
The brain automatically grows new neurons when it encounters high error (confusion). This is similar to how real brains work - when a neuron is "confused" (high error), the brain activates a dormant neuron to help process the new information.

### STDP Learning
Spike-Timing-Dependent Plasticity (STDP) is a learning rule where:
- If neuron A fires *before* neuron B → connection A→B strengthens (Long-Term Potentiation)
- If neuron A fires *after* neuron B → connection A→B weakens (Long-Term Depression)

### AI Dopamine (v3)
The brain uses a pretrained reward model to automatically adjust dopamine levels:
- **Positive sentiment** ("Great job!", "Thanks") → Dopamine increases
- **Negative sentiment** ("No, that's wrong", "Bad") → Dopamine decreases
- **Surprise** (unexpected input) → Dopamine spikes

This creates more biologically accurate learning!

### Hebbian Layer
Quick session-level associations using Hebbian learning ("cells that fire together, wire together"). Helps with short-term memory within a conversation.

## ⚙️ Configuration

```python
# Simple Brain
brain = Brain(
    model_name="my-model",
    vocab_size=10000,
    max_neurons=10000,
    initial_neurons=100,
    lr_stdp=0.001,
    tau_stdp=20.0,
    error_threshold=0.8,
)

# EvolvedChat with LLM
chat = EvolvedChat(
    model="Qwen/Qwen3-0.6B",  # Full HuggingFace path
    brain_path="mybrain.brain",
    max_neurons=10000,
    initial_neurons=100,
)
```

## 📁 File Formats

### .brain file
Contains only the SELSC brain state (weights, neurons, vocab).

### .neuro file
Contains both the brain state AND the LLM model weights (larger file, but fully portable).

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📝 License

MIT License - feel free to use!

---

Made with 🧬 by the Learning Brain Team
