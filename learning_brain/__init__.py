"""
Learning Brain - A Self-Learning Neural Network Package
========================================================

A pip-installable package that provides neurogenesis and continuous learning
for AI models. Like GGUF, but dynamic - it grows new neurons when learning!

Installation:
    pip install learning-brain

Usage:
    from learning_brain import EvolvedChat
    
    # Use with HuggingFace model
    chat = EvolvedChat(model="Qwen/Qwen3-0.6B")
    chat.chat("Hello!")
    
    # Use with local model
    chat = EvolvedChat(chat_model="model.gguf")
    chat.chat("Tell me something")

For CLI:
    python -m learning_brain run qwen
    python -m learning_brain create qwen
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

__version__ = "2.0.0"
__author__ = "Learning Brain Team"

# Import SELSC Engine
from learning_brain.selsc_engine import SELSC_Engine


class EvolvedChat:
    """
    A chat interface that combines an LLM with a growing SELSC brain.
    
    Usage:
        # With HuggingFace model
        chat = EvolvedChat(model="Qwen/Qwen3-0.6B")
        
        # Or with local model
        chat = EvolvedChat(chat_model="model.gguf")
        
        # Chat - brain learns automatically
        response = chat.chat("Hello, how are you?")
        
        # Save brain
        chat.save("my_brain.brain")
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        chat_model: Optional[str] = None,
        brain_path: Optional[str] = None,
        max_neurons: int = 10000,
        initial_neurons: int = 100,
    ):
        """
        Initialize the chat with a model and optional brain.
        
        Args:
            model: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
            chat_model: Path to local GGUF model
            brain_path: Path to .brain file (or creates new one)
            max_neurons: Max neurons for new brain
            initial_neurons: Initial neurons for new brain
        """
        self.model_name = model
        self.chat_model_path = chat_model
        self.brain_path = brain_path
        
        # Load brain
        if brain_path and os.path.exists(brain_path):
            print(f"Loading brain: {brain_path}")
            self._load_brain(brain_path)
        else:
            print("Creating new SELSC brain...")
            self.engine = SELSC_Engine(
                max_neurons=max_neurons,
                initial_neurons=initial_neurons
            )
        
        # Load LLM
        self.llm = None
        self.tokenizer = None
        
        if model:
            self._load_huggingface_model(model)
        elif chat_model:
            print("Note: Local GGUF loading not implemented yet. Use model=...")
    
    def _load_huggingface_model(self, model_name: str):
        """Load a HuggingFace model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            self.model_name = model_name
            print("Model loaded!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_brain(self, path: str):
        """Load brain from file."""
        import numpy as np
        
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        
        self.engine = SELSC_Engine()
        
        # Convert JAX arrays to numpy if needed
        def to_numpy(arr):
            if hasattr(arr, 'tolist'):  # numpy or JAX array
                return np.array(arr)
            return arr
        
        self.engine.weights = to_numpy(bundle['state']['weights'])
        self.engine.V = to_numpy(bundle['state']['V'])
        self.engine.active_mask = to_numpy(bundle['state']['active_mask'])
        self.engine.traces = to_numpy(bundle['state']['traces'])
        self.engine.cum_error = to_numpy(bundle['state']['cum_error'])
        self.engine.vocab = bundle.get('vocab', {})
        self.engine.dopamine = bundle['metadata'].get('dopamine', 1.0)
        self.engine.total_interactions = bundle['metadata'].get('total_interactions', 0)
        self.engine.neurogenesis_events = bundle['metadata'].get('neurogenesis_events', 0)
        
        print(f"Brain loaded - Neurons: {self.engine.info['active_neurons']}")
    
    def chat(self, user_input: str, max_tokens: int = 150) -> str:
        """
        Chat with the AI. The brain will learn from the input.
        
        Args:
            user_input: User's message
            max_tokens: Max tokens to generate
            
        Returns:
            AI's response
        """
        if not self.llm or not self.tokenizer:
            return "No model loaded. Use model=... when creating EvolvedChat."
        
        # Process through SELSC brain (learning)
        tokens = self.tokenizer(user_input, add_special_tokens=False).input_ids
        neurons_grown = 0
        
        import torch
        for token in tokens:
            input_v = torch.zeros(10000)
            input_v[token % 10000] = 1.0
            result = self.engine.step(input_v.cpu().numpy())
            if result['neuron_added']:
                neurons_grown += 1
        
        # Generate response
        messages = [
            {"role": "system", "content": "You are an AI with evolving biological memory that grows."},
            {"role": "user", "content": user_input}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        # Print stats
        print(f"[Brain: {len(tokens)} tokens, +{neurons_grown} neurons, Dop: {self.engine.dopamine:.2f}]")
        
        # Decay dopamine
        self.engine.decay_dopamine()
        
        return response
    
    def save(self, path: Optional[str] = None, include_model: bool = True):
        """Save the brain and optionally the model to a .neuro file.
        
        Args:
            path: Path to save (.neuro extension)
            include_model: If True, saves Qwen model weights too (larger file)
        """
        path = path or self.brain_path or "evolved_brain.neuro"
        
        # Ensure .neuro extension
        if not path.endswith('.neuro'):
            path = path.replace('.brain', '.neuro')
        
        bundle = {
            'state': {
                'weights': self.engine.weights,
                'V': self.engine.V,
                'active_mask': self.engine.active_mask,
                'traces': self.engine.traces,
                'cum_error': self.engine.cum_error,
            },
            'vocab': self.engine.vocab,
            'metadata': {
                'dopamine': self.engine.dopamine,
                'version': '2.0.0',
                'total_interactions': self.engine.total_interactions,
                'neurogenesis_events': self.engine.neurogenesis_events,
                'model_name': self.model_name,
            }
        }
        
        # Optionally include model
        if include_model and self.llm is not None:
            print("Saving model weights (this may take a moment)...")
            bundle['model_weights'] = self.llm.state_dict()
            bundle['model_config'] = {
                'name': self.model_name,
                'config': self.llm.config.to_dict() if hasattr(self.llm.config, 'to_dict') else dict(self.llm.config),
            }
            print("Model weights saved!")
        
        with open(path, 'wb') as f:
            pickle.dump(bundle, f)
        
        print(f"Neuro file saved: {path}")
        return path
    
    @classmethod
    def load_neuro(cls, path: str) -> 'EvolvedChat':
        """Load a .neuro file and return ready-to-use EvolvedChat.
        
        Args:
            path: Path to .neuro file
            
        Returns:
            EvolvedChat instance ready to chat
        """
        print(f"Loading neuro file: {path}")
        
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        
        model_name = bundle['metadata'].get('model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
        
        # Create instance
        chat = cls.__new__(cls)
        chat.model_name = model_name
        chat.brain_path = path
        chat.chat_model_path = None
        
        # Load brain state
        chat.engine = SELSC_Engine()
        chat.engine.weights = bundle['state']['weights']
        chat.engine.V = bundle['state']['V']
        chat.engine.active_mask = bundle['state']['active_mask']
        chat.engine.traces = bundle['state']['traces']
        chat.engine.cum_error = bundle['state']['cum_error']
        chat.engine.vocab = bundle.get('vocab', {})
        chat.engine.dopamine = bundle['metadata'].get('dopamine', 1.0)
        chat.engine.total_interactions = bundle['metadata'].get('total_interactions', 0)
        chat.engine.neurogenesis_events = bundle['metadata'].get('neurogenesis_events', 0)
        
        # Load model if available
        chat.llm = None
        chat.tokenizer = None
        
        if 'model_weights' in bundle:
            print("Loading saved model...")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                chat.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                chat.llm = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                    trust_remote_code=True
                )
                chat.llm.load_state_dict(bundle['model_weights'])
                print("Model loaded from neuro file!")
            except Exception as e:
                print(f"Could not load saved model: {e}")
                print("Loading fresh model instead...")
                chat._load_huggingface_model(model_name)
        else:
            print("No model in neuro file, loading from HuggingFace...")
            chat._load_huggingface_model(model_name)
        
        print(f"Neuro loaded - Neurons: {chat.engine.info['active_neurons']}")
        return chat
    
    @property
    def info(self) -> Dict:
        """Get brain info."""
        return {
            'model': self.model_name,
            'brain_path': self.brain_path,
            **self.engine.info
        }
    
    def __repr__(self):
        return f"EvolvedChat(model={self.model_name}, neurons={self.engine.info['active_neurons']})"


# Simple Brain class (original)
class Brain:
    """A self-learning neural network with neurogenesis and STDP."""
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        model_name: str = "default",
        brain_path: Optional[str] = None,
        vocab_size: int = 10000,
        max_neurons: int = 10000,
        initial_neurons: int = 100,
        lr_stdp: float = 0.001,
        tau_stdp: float = 20.0,
        error_threshold: float = 0.8,
    ):
        self.model_name = model_name
        self.brain_path = brain_path or f"{model_name}.brain"
        
        if os.path.exists(self.brain_path):
            print(f"Loading existing brain from {self.brain_path}...")
            self._load_from_file()
        else:
            print(f"Creating new brain for model: {model_name}")
            self._init_new(
                vocab_size=vocab_size,
                max_neurons=max_neurons,
                initial_neurons=initial_neurons,
                lr_stdp=lr_stdp,
                tau_stdp=tau_stdp,
                error_threshold=error_threshold
            )
    
    def _init_new(self, vocab_size, max_neurons, initial_neurons, lr_stdp, tau_stdp, error_threshold):
        self.config = {
            'vocab_size': vocab_size,
            'max_neurons': max_neurons,
            'initial_neurons': initial_neurons,
            'lr_stdp': lr_stdp,
            'tau_stdp': tau_stdp,
            'error_threshold': error_threshold
        }
        
        self.weights = np.random.randn(max_neurons, max_neurons) * 0.01
        self.V = np.zeros(max_neurons)
        self.active_mask = np.zeros(max_neurons)
        self.active_mask[:initial_neurons] = 1
        self.traces = np.zeros(max_neurons)
        self.cum_error = np.zeros(max_neurons)
        
        self.vocab: Dict[str, int] = {}
        self.learning_history: List[Dict] = []
        self.session_count = 0
        self.total_interactions = 0
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
    
    def _load_from_file(self):
        with open(self.brain_path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.weights = state['weights']
        self.V = state['V']
        self.active_mask = state['active_mask']
        self.traces = state['traces']
        self.cum_error = state['cum_error']
        self.vocab = state['vocab']
        self.learning_history = state.get('learning_history', [])
        self.session_count = state.get('session_count', 0)
        self.total_interactions = state.get('total_interactions', 0)
        self.created_at = state.get('created_at', datetime.now().isoformat())
        self.last_updated = state.get('last_updated', datetime.now().isoformat())
    
    def save(self, path: Optional[str] = None):
        path = path or self.brain_path
        
        state = {
            'config': self.config,
            'weights': self.weights,
            'V': self.V,
            'active_mask': self.active_mask,
            'traces': self.traces,
            'cum_error': self.cum_error,
            'vocab': self.vocab,
            'learning_history': self.learning_history,
            'session_count': self.session_count,
            'total_interactions': self.total_interactions,
            'created_at': self.created_at,
            'last_updated': datetime.now().isoformat(),
            'version': self.VERSION
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Brain saved: {path}")
        return path
    
    @classmethod
    def load(cls, path: str) -> 'Brain':
        brain = cls.__new__(cls)
        brain.model_name = os.path.splitext(os.path.basename(path))[0]
        brain.brain_path = path
        brain._load_from_file()
        return brain
    
    def learn(self, text: str) -> Dict[str, Any]:
        tokens = text.lower().split()
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
        
        max_n = self.config['max_neurons']
        inputs = np.zeros(max_n)
        for t in tokens:
            idx = self.vocab[t]
            if idx < max_n:
                inputs[idx] += 1.0
        
        if np.sum(inputs) > 0:
            inputs = inputs / np.sum(inputs)
        
        v_prev = self.V.copy()
        active_w = self.weights * (
            self.active_mask.reshape(-1, 1) * self.active_mask.reshape(1, -1)
        )
        syn_input = np.dot(active_w[:max_n, :], inputs)
        self.V = self.V + syn_input
        
        spikes = (self.V > 1.0).astype(float) * self.active_mask
        self.V = np.where(spikes > 0, 0.0, self.V)
        
        # Simple STDP
        self.traces = self.traces * np.exp(-1.0 / self.config['tau_stdp']) + inputs
        dw = self.config['lr_stdp'] * np.outer(spikes, self.traces)
        active = self.active_mask.reshape(-1, 1) * self.active_mask.reshape(1, -1)
        self.weights = np.clip(self.weights + dw * active, -1.0, 1.0)
        
        error = np.abs(self.V - v_prev)
        self.cum_error += error * self.active_mask
        
        # Neurogenesis
        grown = False
        active_indices = np.where(self.active_mask == 1)[0]
        if len(active_indices) > 0:
            errors = self.cum_error[active_indices]
            max_error_idx = active_indices[np.argmax(errors)]
            if self.cum_error[max_error_idx] > self.config['error_threshold']:
                inactive = np.where(self.active_mask == 0)[0]
                if len(inactive) > 0:
                    self.active_mask[inactive[0]] = 1
                    self.cum_error[max_error_idx] = 0.0
                    grown = True
        
        self.total_interactions += 1
        self.last_updated = datetime.now().isoformat()
        
        return {
            'tokens': len(tokens),
            'active_neurons': int(np.sum(self.active_mask)),
            'neuron_grown': grown,
            'interactions': self.total_interactions
        }
    
    @property
    def info(self) -> Dict:
        return {
            'model': self.model_name,
            'active_neurons': int(np.sum(self.active_mask)),
            'vocab_size': len(self.vocab),
            'interactions': self.total_interactions,
        }
    
    def __repr__(self):
        return f"Brain(model={self.model_name}, neurons={int(np.sum(self.active_mask))})"


def run(model_name: str = "default", brain_path: Optional[str] = None):
    """Run an interactive session."""
    chat = EvolvedChat(model=model_name, brain_path=brain_path)
    
    print("\n" + "="*50)
    print(f"Evolved Chat - {model_name}")
    print("="*50)
    print(f"Info: {chat.info}")
    print("\nCommands:")
    print("  save - Save and exit")
    print("  info - Show brain info")
    print("  quit - Exit without saving")
    print("-"*50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ('save', 'quit'):
                if user_input.lower() == 'save':
                    chat.save()
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'info':
                for k, v in chat.info.items():
                    print(f"  {k}: {v}")
            
            elif user_input:
                response = chat.chat(user_input)
                print(f"\nAI: {response}")
        
        except KeyboardInterrupt:
            print("\n\nSaving...")
            chat.save()
            break


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nCommands:")
        print("  python -m learning_brain run <model_name>")
        print("  python -m learning_brain create <model_name>")
        return
    
    command = sys.argv[1]
    
    if command == "run":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-0.6B"
        run(model_name)
    
    elif command == "create":
        brain = Brain(sys.argv[2] if len(sys.argv) > 2 else "default")
        brain.save()
        print(f"Created brain for {brain.model_name}")


if __name__ == "__main__":
    main()
