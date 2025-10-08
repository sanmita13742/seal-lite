"""
T4-Optimized Grader for SEAL
Uses local models instead of OpenAI API
Lightweight and efficient for T4 GPU
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List, Dict


class T4Grader:
    """
    Local grading using small language models.
    Optimized for T4 GPU memory constraints.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize T4 grader.
        
        Args:
            model_name: Small model for grading (default: Llama 1B)
            device: Device to use
        """
        self.device = device
        self.model_name = model_name
        
        print(f"[T4Grader] Loading grader model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model (use 8-bit for memory efficiency)
        from transformers import BitsAndBytesConfig
        
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
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
                torch_dtype=torch.float32,
            )
        
        self.model.eval()
        print(f"[T4Grader] Grader loaded successfully")
    
    def grade_yes_no(
        self,
        question: str,
        answer: str,
        context: str = ""
    ) -> bool:
        """
        Grade a yes/no question.
        
        Args:
            question: Question text
            answer: Provided answer
            context: Additional context (optional)
        
        Returns:
            True if correct, False otherwise
        """
        # Create grading prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a precise grading assistant. Answer ONLY with 'yes' or 'no'.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}
Answer: {answer}
{f"Context: {context}" if context else ""}

Is this answer correct? Reply ONLY with 'yes' or 'no'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()
        
        # Parse yes/no
        return 'yes' in response[:10]
    
    def grade_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[str] = None
    ) -> List[bool]:
        """
        Grade multiple questions in batch.
        
        Args:
            questions: List of questions
            answers: List of answers
            contexts: List of contexts (optional)
        
        Returns:
            List of boolean grades
        """
        if contexts is None:
            contexts = [""] * len(questions)
        
        results = []
        for q, a, c in zip(questions, answers, contexts):
            grade = self.grade_yes_no(q, a, c)
            results.append(grade)
        
        return results
    
    def exact_match(
        self,
        predicted: str,
        ground_truth: str,
        normalize: bool = True
    ) -> bool:
        """
        Check exact match between predicted and ground truth.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            normalize: Whether to normalize (lowercase, strip)
        
        Returns:
            True if exact match
        """
        if normalize:
            predicted = predicted.strip().lower()
            ground_truth = ground_truth.strip().lower()
        
        return predicted == ground_truth
    
    def contains_match(
        self,
        predicted: str,
        ground_truth: str,
        normalize: bool = True
    ) -> bool:
        """
        Check if ground truth is contained in prediction.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            normalize: Whether to normalize
        
        Returns:
            True if ground truth in predicted
        """
        if normalize:
            predicted = predicted.strip().lower()
            ground_truth = ground_truth.strip().lower()
        
        return ground_truth in predicted
    
    def cleanup(self):
        """Clean up resources."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        print("[T4Grader] Resources cleaned up")


# Compatibility function for existing code
def grade_with_local_model(
    questions: List[str],
    answers: List[str],
    grader: T4Grader = None,
    **kwargs
) -> List[bool]:
    """
    Grade questions using local model (replaces OpenAI grading).
    
    Args:
        questions: List of questions
        answers: List of answers
        grader: T4Grader instance (will create if None)
        **kwargs: Additional arguments
    
    Returns:
        List of boolean grades
    """
    if grader is None:
        grader = T4Grader()
    
    return grader.grade_batch(questions, answers)
