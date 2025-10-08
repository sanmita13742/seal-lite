"""
Ollama-based inference engine for local Llama models.
Replaces vLLM for laptop/CPU usage.
"""
from typing import Dict, List, Optional, Tuple
import requests
import json
from transformers import PreTrainedTokenizer, AutoTokenizer


class OllamaSamplingParams:
    """Sampling parameters for Ollama API."""
    def __init__(self, max_tokens: int, temperature: float = 0.0, n: int = 1):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n


class OllamaEngine:
    """Lightweight Ollama engine wrapper."""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()
        
    def generate(self, prompt: str, sampling_params: OllamaSamplingParams) -> List[str]:
        """Generate completions using Ollama API."""
        outputs = []
        
        for _ in range(sampling_params.n):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": sampling_params.temperature,
                            "num_predict": sampling_params.max_tokens,
                        }
                    },
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    outputs.append(result.get("response", ""))
                else:
                    print(f"Ollama API error: {response.status_code}")
                    outputs.append("")
                    
            except Exception as e:
                print(f"Error calling Ollama: {e}")
                outputs.append("")
                
        return outputs


def get_sampling_params(
    tokenizer: PreTrainedTokenizer,
    num_tokens: int,
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
) -> OllamaSamplingParams:
    """Create sampling parameters."""
    max_new_tokens = max_tokens - num_tokens
    return OllamaSamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        n=n
    )


def initialize_engine(
    model: str,
    enforce_eager: bool = False,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
    quantization: Optional[str] = None,
    lora_repo: Optional[str] = None,
    lora_target_modules: Optional[List[str]] = None,
    ollama_base_url: str = "http://localhost:11434"
) -> OllamaEngine:
    """Initialize the Ollama engine."""
    print(f"Initializing Ollama engine with model: {model}")
    print(f"Note: LoRA is disabled for Ollama (enable_lora={enable_lora} ignored)")
    return OllamaEngine(model=model, base_url=ollama_base_url)


def process_requests(
    engine: OllamaEngine, 
    test_prompts: List[Tuple[str, OllamaSamplingParams, Optional[any], str]]
) -> Dict[str, List[str]]:
    """Process prompts with Ollama engine."""
    all_outputs: Dict[str, List[str]] = {}
    
    print(f"Processing {len(test_prompts)} prompts with Ollama...")
    
    for i, (prompt, sampling_param, lora_request, idx) in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] Processing task {idx}...")
        
        # Clean prompt if needed
        if "<|begin_of_text|>" in prompt:
            find_start = prompt.find("<|begin_of_text|>") + len("<|begin_of_text|>")
            prompt = prompt[find_start:]
        
        # Generate outputs
        texts = engine.generate(prompt, sampling_param)
        all_outputs[idx] = texts
        
    return all_outputs
