# 🧠 Learning Brain

> A self-learning neural network that grows new neurons! Like GGUF, but dynamic.

**Learning Brain** is both a Python pip package AND a HuggingFace model. This README explains the full project - both the model we trained and the package we built around it.

---

## 🎯 The Big Picture: Why We Built This

### The Problem with Current AI

Current Large Language Models (LLMs) like GPT-4, Claude, and Llama are **static**. They learn everything during training, and once trained, they can't learn anything new. They're frozen in time.

When you chat with ChatGPT, it can't:
- Remember things from previous conversations (beyond the current session)
- Learn from your corrections
- Grow new neural pathways when it encounters something it doesn't understand
- Adapt its learning rate based on whether it's doing well or poorly

### Our Solution: A Brain That Grows

We built a brain that:
1. **Learns continuously** - It can learn during inference, not just training
2. **Grows new neurons** - When confused, it literally creates new neural pathways
3. **Uses dopamine** - Like real brains, it has a reward system that modulates learning
4. **Detects sentiment** - It knows when you're praising or correcting it
5. **Makes associations** - Things learned together stay together (Hebbian learning)

---

## 🤖 The HuggingFace Model

**Model Page:** https://huggingface.co/Specialgfhdhdh/learning-qwen

This is a **Qwen2.5-0.5B** model fine-tuned with our SELSC (Spike-timing-dependent Elastic Weight Consolidation Self-Consolidation) brain architecture.

### What's Special About This Model?

Unlike regular LLMs, our model has:

1. **A Growing Neural Brain** - The model has a companion brain that grows neurons when learning new things

2. **Dopamine Modulation** - Learning rate is modulated by a simulated dopamine system:
   - High dopamine = fast learning (exciting information!)
   - Low dopamine = slow learning (boring/repetitive info)

3. **Hebbian Associations** - "Cells that fire together, wire together" - the brain forms quick associations between concepts learned in the same session

4. **Neurogenesis** - When the brain encounters high error (confusion), it literally activates dormant neurons to help process the new information

---

## 🧠 How It Works: The Science

### 1. Neurogenesis: Growing New Neurons

In real brains, neurogenesis (creating new neurons) happens in the hippocampus. Our artificial neurogenesis works similarly:

```python
# When a neuron has high cumulative error (confusion)
if self.cum_error[max_error_idx] > self.error_threshold:
    # Activate a dormant neuron!
    self.active_mask[inactive_neuron] = 1
```

**Why this matters:** It means the brain can expand its capacity when needed. If it encounters something completely new that confuses its existing neurons, it grows new ones to handle it.

### 2. STDP: Spike-Timing-Dependent Plasticity

STDP is how real neurons learn. The key insight:

- **If neuron A fires BEFORE neuron B** → Connection A→B **strengthens** (Long-Term Potentiation)
- **If neuron A fires AFTER neuron B** → Connection A→B **weakens** (Long-Term Depression)

```python
# STDP in our code
dw = self.lr_stdp * np.outer(spikes, self.traces)
self.weights = np.clip(self.weights + dw, -1.0, 1.0)
```

This creates causal relationships in the neural network. If A consistently leads to B, the connection strengthens.

### 3. Dopamine: The Reward Signal

In real brains, dopamine is a neurotransmitter that signals "this is good!" or "this is bad!" It's released when we:
- Eat food → dopamine spike
- Get praised → dopamine spike
- Make a mistake → dopamine decreases

Our brain simulates this:

```python
def apply_dopamine(self, reward: float):
    # Reward modulates learning rate
    multiplier = 1.0 + reward * 0.5
    self.dopamine = np.clip(self.dopamine * multiplier, 0.1, 3.0)
```

High dopamine = high learning rate = the brain is in "learning mode"
Low dopamine = low learning rate = the brain is in "stable mode"

### 4. AI-Modulated Dopamine (v3)

In version 3, we added an **AI that decides the dopamine level**. The TinyRewardModel looks at what's being learned and decides:

- Is this surprising? → Dopamine UP
- Is this repetitive? → Dopamine DOWN
- Is the user praising? → Dopamine UP
- Is the user correcting? → Dopamine DOWN

### 5. Hebbian Learning: Quick Associations

Donald Hebbian's famous principle: **"Neurons that fire together, wire together"**

```python
def learn(self, word_idx: int, context_words: List[int]):
    # Strengthen associations between words in the same context
    for ctx_word in context_words:
        self.associations[word_idx][ctx_word] += 0.1
```

This creates quick session-level associations. If you teach it "Apple is a fruit" and then "Banana is a fruit", it quickly associates Apple ↔ Banana ↔ fruit.

### 6. Sentiment Detection

The brain can detect if you're praising or correcting it:

```python
praise_words = ['good', 'great', 'awesome', 'thanks', 'correct', 'yes']
correction_words = ['wrong', 'no', 'incorrect', 'but', 'actually', 'mistake']
```

When it detects praise → dopamine increases → learns faster!
When it detects correction → dopamine decreases → tries harder next time!

---

## 📦 The Python Package

While the core brain technology lives in the HuggingFace model, we also built a **pip-installable Python package** that lets you use similar brain-like features with ANY model.

### Installation

```bash
pip install learning-brain
```

### Usage

```python
from learning_brain import Brain

# Create a learning brain
brain = Brain("my-model")

# Teach it something
result = brain.learn("The sky is blue")
print(result)
# {'tokens': 4, 'active_neurons': 100, 'neuron_grown': False, 'interactions': 1}

# With an LLM
from learning_brain import EvolvedChat
chat = EvolvedChat(model="Qwen/Qwen3-0.6B")
chat.chat("Hello! Teach me about physics.")
```

---

## 🔬 Technical Details

### Architecture Overview

```
User Input → Tokenize → SELSC Brain → LLM → Response
                ↓
         [Neurogenesis]
                ↓
         [STDP Learning]  
                ↓
         [Dopamine Update]
                ↓
         [Hebbian Associations]
                ↓
         [Save State]
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_neurons | 10000 | Maximum neurons the brain can grow |
| initial_neurons | 100 | Starting neurons |
| base_lr_stdp | 0.05 | Base STDP learning rate |
| tau_stdp | 20.0 | STDP time constant |
| error_threshold | 0.02 | Error threshold for neurogenesis |
| dopamine | 1.0 | Initial dopamine level |

### File Formats

- **.brain** - Just the brain state (weights, neurons, vocab)
- **.neuro** - Brain state + LLM model weights (fully portable)

---

## 🎓 Why This Matters

### Current AI Limitations

1. **No Continuous Learning** - LLMs can't learn after training
2. **Static Capacity** - Can't grow to handle new information
3. **No Reward System** - Don't know if they're doing well
4. **No Session Memory** - Can't form quick associations within a session

### How We Address These

| Limitation | Our Solution |
|------------|---------------|
| No continuous learning | ✅ Brain learns during inference |
| Static capacity | ✅ Neurogenesis grows new neurons |
| No reward system | ✅ Dopamine modulates learning rate |
| No session memory | ✅ Hebbian layer quick associations |

---

## 🚀 Future Directions

This is just the beginning! Possible extensions:

1. **Multi-modal neurogenesis** - Grow new visual/sensory pathways
2. **Long-term Hebbian memory** - Persist associations across sessions
3. **Curiosity-driven learning** - Seek out novel information
4. **Emotional state** - Model mood variations
5. **Sleep consolidation** - Simulate memory consolidation during "downtime"

---

## 📚 References

- **STDP:** Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and neuronal type.
- **Neurogenesis:** Gage, F. H. (2000). Mammalian neural stem cells.
- **Hebbian Learning:** Hebb, D. O. (1949). The Organization of Behavior.
- **Dopamine Learning:** Schultz, W. (1998). Predictive Reward Signal of Dopamine Neurons.

---

## 🤝 Contributing

This is an open research project! Contributions welcome:
- Improve the neurogenesis algorithm
- Add new dopamine modulation strategies
- Better sentiment detection
- Performance optimizations

---

## 📝 License

MIT License - free to use!

---

## 🔗 Links

- **HuggingFace Model:** https://huggingface.co/Specialgfhdhdh/learning-qwen
- **PyPI Package:** https://pypi.org/project/learning-brain/
- **GitHub:** https://github.com/jihvijhojhviihogyuvi/learning_brain

---

Made with 🧬 by the Learning Brain Team
