"""
T4-Optimized Inference Engine for SEAL
Uses HuggingFace Transformers directly (no vLLM)
Supports batch processing and LoRA adapters
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import List, Dict, Optional, Union
import gc


class T4Engine:
    """
    T4-optimized inference engine using HuggingFace Transformers.
    Supports 8-bit quantization for memory efficiency.
    """
    
    def __init__(
        self,
        model_name: str,
        use_8bit: bool = True,
        max_length: int = 2048,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize T4 engine.
        
        Args:
            model_name: HuggingFace model name or path
            use_8bit: Use 8-bit quantization (saves ~50% VRAM)
            max_length: Maximum sequence length
            device: Device to use
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.use_8bit = use_8bit
        
        print(f"[T4Engine] Loading model: {model_name}")
        print(f"[T4Engine] 8-bit quantization: {use_8bit}")
        print(f"[T4Engine] Device: {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optional 8-bit quantization
        if use_8bit and device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
        
        self.model.eval()
        self.loaded_adapters = {}
        
        print(f"[T4Engine] Model loaded successfully")
        print(f"[T4Engine] Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    def load_adapter(self, adapter_path: str, adapter_name: str = "default"):
        """Load a LoRA adapter."""
        if adapter_name in self.loaded_adapters:
            print(f"[T4Engine] Adapter '{adapter_name}' already loaded")
            return
        
        print(f"[T4Engine] Loading adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=False
        )
        self.loaded_adapters[adapter_name] = adapter_path
        print(f"[T4Engine] Adapter loaded: {adapter_name}")
    
    def unload_adapter(self, adapter_name: str = "default"):
        """Unload a LoRA adapter."""
        if adapter_name not in self.loaded_adapters:
            print(f"[T4Engine] Adapter '{adapter_name}' not loaded")
            return
        
        # For PEFT models, we need to switch back to base model
        if hasattr(self.model, 'disable_adapter'):
            self.model.disable_adapter()
        
        del self.loaded_adapters[adapter_name]
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[T4Engine] Adapter unloaded: {adapter_name}")
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        adapter_name: Optional[str] = None,
    ) -> List[str]:
        """
        Generate completions for prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences per prompt
            do_sample: Whether to sample (False = greedy)
            adapter_name: Name of adapter to use (if any)
        
        Returns:
            List of generated texts
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Set adapter if specified
        if adapter_name and adapter_name in self.loaded_adapters:
            if hasattr(self.model, 'set_adapter'):
                self.model.set_adapter(adapter_name)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove input prompt from output
            input_length = inputs['input_ids'][i // num_return_sequences].shape[0]
            generated_ids = output[input_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Generate in batches to manage memory."""
        all_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            outputs = self.generate(batch, **kwargs)
            all_outputs.extend(outputs)
            
            # Clear cache between batches
            gc.collect()
            torch.cuda.empty_cache()
        
        return all_outputs
    
    def cleanup(self):
        """Clean up resources."""
        del self.model
        del self.tokenizer
        self.loaded_adapters = {}
        gc.collect()
        torch.cuda.empty_cache()
        print("[T4Engine] Resources cleaned up")


def initialize_engine(
    model: str,
    use_8bit: bool = True,
    max_length: int = 2048,
    **kwargs
) -> T4Engine:
    """
    Initialize T4 engine (compatible with original interface).
    
    Args:
        model: Model name or path
        use_8bit: Use 8-bit quantization
        max_length: Maximum sequence length
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        T4Engine instance
    """
    return T4Engine(
        model_name=model,
        use_8bit=use_8bit,
        max_length=max_length
    )


def process_requests(
    engine: T4Engine,
    requests: List[Dict],
    adapter_path: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Process requests (compatible with original interface).
    
    Args:
        engine: T4Engine instance
        requests: List of request dicts with 'prompt' key
        adapter_path: Path to LoRA adapter (if any)
        **kwargs: Additional generation parameters
    
    Returns:
        List of generated texts
    """
    # Load adapter if provided
    if adapter_path:
        engine.load_adapter(adapter_path, adapter_name="current")
    
    # Extract prompts
    prompts = [req['prompt'] for req in requests]
    
    # Generate
    outputs = engine.batch_generate(prompts, **kwargs)
    
    # Unload adapter if loaded
    if adapter_path:
        engine.unload_adapter("current")
    
    return outputs
