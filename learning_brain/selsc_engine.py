"""
SELSC Engine v3 - The Evolved Brain with AI Dopamine Control
=============================================================

Features:
- Neurogenesis (grows new neurons when confused)
- AI-modulated dopamine (uses tiny reward model to decide dopamine)
- Hebbian layer for quick session associations
- Sentiment detection (praise/correction detection)

Uses NumPy + optional HuggingFace hub for pretrained reward model.
"""

import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional


class TinyRewardModel:
    """
    Tiny AI that decides dopamine based on LLM hidden states.
    Loaded from HuggingFace or uses fallback heuristic.
    """
    
    def __init__(self, use_pretrained: bool = True):
        self.model = None
        self.projection_matrix = None
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            self._load_pretrained()
    
    def _load_pretrained(self):
        """Load pretrained reward model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            
            # Download brain.bundle from HF
            path = hf_hub_download(
                repo_id="Specialgfhdhdh/learning-qwen",
                filename="brain.bundle"
            )
            
            with open(path, 'rb') as f:
                bundle = pickle.load(f)
            
            # Extract reward model weights
            self.model = bundle.get('tiny_reward_weights', None)
            self.projection_matrix = bundle.get('projection_matrix', None)
            
            if self.model is not None:
                print("Loaded pretrained dopamine AI from HuggingFace!")
            else:
                print("No reward model in bundle, using fallback")
                
        except Exception as e:
            print(f"Could not load pretrained reward model: {e}")
            print("Using fallback dopamine calculation")
            self.use_pretrained = False
    
    def compute_reward(self, hidden_state: np.ndarray) -> float:
        """Compute reward signal from LLM hidden state."""
        if self.model is not None and 'head.weight' in self.model:
            weight = self.model['head.weight']
            bias = self.model['head.bias']
            
            # Convert torch tensors if needed
            if hasattr(weight, 'numpy'):
                weight = weight.numpy()
            if hasattr(bias, 'numpy'):
                bias = bias.numpy()
            
            reward = float(np.dot(weight, hidden_state) + bias)
            return np.clip(reward, -1.0, 1.0)
        else:
            return float(np.clip(np.std(hidden_state) - 0.5, -1.0, 1.0))
    
    def compute_reward_from_brain(self, brain_state: np.ndarray) -> float:
        """Compute reward from brain state using projection matrix."""
        if self.projection_matrix is not None:
            hidden = np.dot(self.projection_matrix.T, brain_state)
            return self.compute_reward(hidden)
        return 0.0


class HebbianLayer:
    """Hebbian layer for quick session associations."""
    
    def __init__(self, vocab_size: int = 10000):
        self.associations = {}
        self.session_memory = []
        self.decay_rate = 0.9
    
    def learn(self, word_idx: int, context_words: List[int]):
        """Learn associations between words in context."""
        if word_idx not in self.associations:
            self.associations[word_idx] = {}
        
        for ctx_word in context_words:
            if ctx_word != word_idx:
                if ctx_word not in self.associations[word_idx]:
                    self.associations[word_idx][ctx_word] = 0.0
                self.associations[word_idx][ctx_word] += 0.1
        
        self.session_memory.append(word_idx)
        if len(self.session_memory) > 100:
            self.session_memory.pop(0)
    
    def get_associations(self, word_idx: int) -> List[int]:
        """Get associated word indices."""
        if word_idx not in self.associations:
            return []
        assoc = self.associations[word_idx]
        return sorted(assoc.keys(), key=lambda x: assoc[x], reverse=True)[:10]
    
    def decay(self):
        """Decay old associations."""
        for word_idx in self.associations:
            for assoc_idx in self.associations[word_idx]:
                self.associations[word_idx][assoc_idx] *= self.decay_rate
            
            weak = [k for k, v in self.associations[word_idx].items() if v < 0.01]
            for k in weak:
                del self.associations[word_idx][k]


class SELSC_Engine:
    """SELSC Engine v3 with AI-modulated dopamine!"""
    
    VERSION = "3.0.0"
    
    def __init__(
        self,
        max_neurons: int = 10000,
        initial_neurons: int = 100,
        base_lr_stdp: float = 0.05,
        tau_stdp: float = 20.0,
        error_threshold: float = 0.02,
        dopamine: float = 1.0,
        use_pretrained_reward: bool = True,
    ):
        self.max_neurons = max_neurons
        self.initial_neurons = initial_neurons
        self.base_lr_stdp = base_lr_stdp
        self.tau_stdp = tau_stdp
        self.error_threshold = error_threshold
        self.dopamine = dopamine
        
        # AI reward model
        self.reward_model = TinyRewardModel(use_pretrained=use_pretrained_reward)
        
        # Hebbian layer
        self.hebbian = HebbianLayer(vocab_size=max_neurons)
        
        self._init_network()
        
        self.vocab: Dict[str, int] = {}
        
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.total_interactions = 0
        self.neurogenesis_events = 0
        
    def _init_network(self):
        """Initialize the neural network."""
        self.weights = np.random.randn(self.max_neurons, self.max_neurons) * 0.01
        self.V = np.zeros(self.max_neurons)
        self.active_mask = np.zeros(self.max_neurons)
        self.active_mask[:self.initial_neurons] = 1
        self.traces = np.zeros(self.max_neurons)
        self.cum_error = np.zeros(self.max_neurons)
        
    @property
    def lr_stdp(self) -> float:
        """Learning rate modulated by dopamine."""
        return np.clip(self.base_lr_stdp * self.dopamine, 0.001, 0.25)
    
    def apply_dopamine(self, reward: float):
        """Apply dopamine reward signal from AI model."""
        ai_reward = self.reward_model.compute_reward_from_brain(self.V)
        combined = (reward + ai_reward) / 2.0
        multiplier = 1.0 + combined * 0.5
        self.dopamine = np.clip(self.dopamine * multiplier, 0.1, 3.0)
        
    def apply_reward(self, reward: float):
        """Apply explicit reward (1.0 = praise, -1.0 = correction)."""
        self.apply_dopamine(reward)
        
    def decay_dopamine(self):
        """Decay dopamine back to baseline."""
        self.dopamine = self.dopamine * 0.85 + 1.0 * 0.15
        self.hebbian.decay()
        
    def detect_sentiment(self, text: str) -> float:
        """Detect praise or correction in text. Returns: 1.0=praise, -1.0=correction, 0.0=neutral"""
        text_lower = text.lower()
        
        praise_words = ['good', 'great', 'awesome', 'perfect', 'thanks', 'thank you', 
                       'correct', 'yes', 'exactly', 'love', 'nice', 'well done']
        correction_words = ['wrong', 'no', 'incorrect', 'actually', 'but', 'however',
                           'mistake', 'error', 'bad', 'not right', 'sorry']
        
        praise_count = sum(1 for w in praise_words if w in text_lower)
        correction_count = sum(1 for w in correction_words if w in text_lower)
        
        if praise_count > correction_count:
            return 0.5
        elif correction_count > praise_count:
            return -0.5
        return 0.0
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        words = text.lower().split()
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        return [self.vocab.get(w, 0) for w in words]
    
    def step(self, input_spikes: np.ndarray, hidden_state: np.ndarray = None) -> Dict[str, Any]:
        """Process one step through the network."""
        v_prev = self.V.copy()
        
        active_weights = self.weights * np.outer(self.active_mask, self.active_mask)
        v_new = self.V + np.dot(active_weights, input_spikes)
        
        spikes = (v_new > 1.0).astype(float) * self.active_mask
        
        # AI-modulated dopamine
        if hidden_state is not None:
            ai_reward = self.reward_model.compute_reward(hidden_state)
            self.dopamine = np.clip(self.dopamine + ai_reward * 0.2, 0.1, 3.0)
        else:
            surprise = np.max(np.abs(v_new - v_prev)) * 15.0
            self.dopamine = np.clip(self.dopamine + surprise * 0.1, 0.1, 3.0)
        
        # STDP learning
        self.traces = self.traces * np.exp(-1.0 / self.tau_stdp) + input_spikes
        dw = self.lr_stdp * np.outer(spikes, self.traces)
        self.weights = np.clip(self.weights + dw, -1.0, 1.0)
        
        # Track error
        self.cum_error += np.abs(v_new - v_prev) * self.active_mask
        
        # Neurogenesis
        added = False
        active_indices = np.where(self.active_mask == 1)[0]
        if len(active_indices) > 0:
            errors = self.cum_error[active_indices]
            max_error_idx = active_indices[np.argmax(errors)]
            if self.cum_error[max_error_idx] > self.error_threshold:
                inactive = np.where(self.active_mask == 0)[0]
                if len(inactive) > 0:
                    self.active_mask[inactive[0]] = 1
                    self.cum_error[max_error_idx] = 0.0
                    added = True
                    self.neurogenesis_events += 1
        
        self.V = np.where(spikes > 0, 0.0, v_new) * self.active_mask
        
        self.total_interactions += 1
        self.last_updated = datetime.now().isoformat()
        
        return {
            'V': self.V.copy(),
            'spikes': spikes,
            'neuron_added': added,
            'dopamine': self.dopamine
        }
    
    def process_text(self, text: str, hidden_state: np.ndarray = None) -> Dict[str, Any]:
        """Process text through the network."""
        tokens = self.tokenize(text)
        
        # Learn hebbian associations
        for i, token in enumerate(tokens):
            context = tokens[max(0, i-5):i] + tokens[i+1:min(len(tokens), i+6)]
            self.hebbian.learn(token, context)
        
        neurons_added = 0
        for token in tokens:
            input_v = np.zeros(self.max_neurons)
            input_v[token % self.max_neurons] = 1.0
            result = self.step(input_v, hidden_state)
            if result['neuron_added']:
                neurons_added += 1
        
        # Detect sentiment
        sentiment = self.detect_sentiment(text)
        if sentiment != 0.0:
            self.apply_dopamine(sentiment)
        
        return {
            'tokens_processed': len(tokens),
            'neurons_added': neurons_added,
            'active_neurons': int(np.sum(self.active_mask)),
            'dopamine': self.dopamine,
            'sentiment': sentiment
        }
    
    def get_state(self) -> Dict:
        """Get state for saving."""
        return {
            'state': {
                'weights': self.weights,
                'V': self.V,
                'active_mask': self.active_mask,
                'traces': self.traces,
                'cum_error': self.cum_error,
            },
            'vocab': self.vocab,
            'hebbian': self.hebbian.associations,
            'metadata': {
                'dopamine': self.dopamine,
                'version': self.VERSION,
                'created_at': self.created_at,
                'last_updated': self.last_updated,
                'total_interactions': self.total_interactions,
                'neurogenesis_events': self.neurogenesis_events,
            }
        }
    
    def save(self, path: str):
        """Save the brain to file."""
        state = self.get_state()
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Brain saved: {path}")
        
    @classmethod
    def load(cls, path: str) -> 'SELSC_Engine':
        """Load brain from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        engine = cls()
        engine.weights = state['state']['weights']
        engine.V = state['state']['V']
        engine.active_mask = state['state']['active_mask']
        engine.traces = state['state']['traces']
        engine.cum_error = state['state']['cum_error']
        engine.vocab = state['vocab']
        engine.hebbian.associations = state.get('hebbian', {})
        engine.dopamine = state['metadata']['dopamine']
        engine.created_at = state['metadata']['created_at']
        engine.last_updated = state['metadata']['last_updated']
        engine.total_interactions = state['metadata']['total_interactions']
        engine.neurogenesis_events = state['metadata']['neurogenesis_events']
        
        print(f"Brain loaded: {path}")
        return engine
    
    @property
    def info(self) -> Dict:
        """Get brain info."""
        return {
            'version': self.VERSION,
            'active_neurons': int(np.sum(self.active_mask)),
            'max_neurons': self.max_neurons,
            'dopamine': self.dopamine,
            'lr_stdp': self.lr_stdp,
            'total_interactions': self.total_interactions,
            'neurogenesis_events': self.neurogenesis_events,
            'vocab_size': len(self.vocab),
            'hebbian_associations': len(self.hebbian.associations),
            'ai_reward_model': self.reward_model.use_pretrained,
        }


def create_selsc_brain(
    name: str = "selsc_brain",
    max_neurons: int = 10000,
    initial_neurons: int = 100,
    path: Optional[str] = None
) -> SELSC_Engine:
    """Create a new SELSC brain."""
    brain = SELSC_Engine(
        max_neurons=max_neurons,
        initial_neurons=initial_neurons
    )
    if path:
        brain.save(path)
    return brain


if __name__ == "__main__":
    print("=== SELSC Engine v3 Demo ===")
    
    brain = create_selsc_brain("demo", max_neurons=1000, initial_neurons=20)
    
    texts = [
        "hello world",
        "artificial intelligence is great",
        "no, that's actually wrong",
    ]
    
    print("\n--- Learning Phase ---")
    for text in texts:
        result = brain.process_text(text)
        print(f"Learned: {text}")
        print(f"  Neurons: {result['active_neurons']}, Dopamine: {result['dopamine']:.2f}, Sentiment: {result['sentiment']}")
    
    print("\n--- Final State ---")
    print(brain.info)
    
    brain.save("selsc_demo.brain")
    print("\nDemo complete!")
