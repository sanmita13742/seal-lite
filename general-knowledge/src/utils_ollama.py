"""
Ollama-based utilities for general knowledge tasks.
Replaces OpenAI API calls with local Ollama.
"""
import requests
import logging
import time
import re
from typing import Any, Dict, List, Optional


# Templates (same as original)
SQUAD_ANSWER_TEMPLATE_BASE = (
    "Let's answer a question directly and concisely.\n"
    "Question: {question}\n"
    "Answer:\n"
)

SQUAD_ANSWER_TEMPLATE_BASE_COT = (
    "Let's think step by step and then answer the question directly and concisely. "
    "Let's first give reasoning under \"Reasoning:\" and then the final answer under \"Final answer:\".\n"
    "Question: {question}\n"
    "Reasoning:"
)

SQUAD_GRADE_TEMPLATE = (
    "You are a grading assistant. Your job is to determine whether a student's answer "
    "correctly answers the question based solely on the provided gold answer. "
    "Do not use any outside knowledge. The student answer can include additional information, "
    "but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. "
    "Respond ONLY with 'yes' or 'no'.\n\n"
    "Question: {question}\nGold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'."
)

TRAINING_SEQUENCE_TEMPLATE = "{title}\n{completion_text}"


class OllamaClient:
    """Simple Ollama API client."""
    
    def __init__(self, model: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()
        logging.info(f"Initialized Ollama client: {model} @ {base_url}")
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """Generate completion from Ollama."""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logging.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logging.error(f"Error calling Ollama: {e}")
            return ""
    
    def generate_batch(self, prompts: List[str], temperature: float = 0.0, max_tokens: int = 512) -> List[str]:
        """Generate completions for batch of prompts."""
        return [self.generate(p, temperature, max_tokens) for p in prompts]


# Global client
_ollama_client: Optional[OllamaClient] = None


def init_ollama_client(model: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
    """Initialize global Ollama client."""
    global _ollama_client
    _ollama_client = OllamaClient(model=model, base_url=base_url)
    return _ollama_client


def get_ollama_client() -> OllamaClient:
    """Get global Ollama client (lazy init)."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = init_ollama_client()
    return _ollama_client


# Answer formatting
def format_answer_prompts(
    q_batch: List[Dict[str, str]], 
    instruct_model: bool = False, 
    chain_of_thought: bool = False
) -> List[str]:
    """Format answer prompts."""
    template = SQUAD_ANSWER_TEMPLATE_BASE_COT if chain_of_thought else SQUAD_ANSWER_TEMPLATE_BASE
    return [template.format(question=q["question"]) for q in q_batch]


def format_grade_prompts(q_batch: List[Dict[str, str]], preds: List[str]) -> List[str]:
    """Format grading prompts."""
    return [
        SQUAD_GRADE_TEMPLATE.format(
            question=q["question"],
            gold=q["answer"],
            pred=p.strip(),
        )
        for q, p in zip(q_batch, preds)
    ]


# Answer extraction
_final_ans_re = re.compile(
    r"(?:^|\n)\s*final\s*answer\s*[:\-]\s*(.*)\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def extract_final_answer(text: str) -> str:
    """Extract final answer from CoT response."""
    if not text:
        return "idk"
    m = _final_ans_re.search(text.strip())
    return (m.group(1).strip() if m else "idk").strip()


# Grading
_yes_re = re.compile(r"\b(yes)\b", re.I)
_no_re = re.compile(r"\b(no)\b", re.I)


def parse_yes_no(text: str) -> bool:
    """Parse yes/no from text."""
    if _yes_re.search(text) and not _no_re.search(text):
        return True
    return False


def grade_with_ollama(prompts: List[str]) -> List[bool]:
    """Grade answers using Ollama."""
    client = get_ollama_client()
    verdicts = []
    
    for i, prompt in enumerate(prompts):
        logging.debug(f"Grading {i+1}/{len(prompts)}...")
        response = client.generate(prompt, temperature=0.0, max_tokens=10)
        verdict = parse_yes_no(response)
        verdicts.append(verdict)
        time.sleep(0.1)  # Small delay to avoid overwhelming Ollama
    
    return verdicts


# Training data utilities
MAX_TRAIN_SEQS_PER_COMPLETION = 30


def build_train_sequences(
    completion_raw: str,
    context: str,
    title: str,
    *,
    split_newlines: bool = False,
    add_context: bool = True,
) -> List[str]:
    """
    Build training sequences from completion text.
    """
    sequences = []
    
    if split_newlines:
        lines = [l.strip() for l in completion_raw.split('\n') if l.strip()]
        for line in lines[:MAX_TRAIN_SEQS_PER_COMPLETION]:
            if add_context:
                seq = f"{title}\n{context}\n{line}"
            else:
                seq = TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=line)
            sequences.append(seq)
    else:
        if add_context:
            seq = f"{title}\n{context}\n{completion_raw}"
        else:
            seq = TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=completion_raw)
        sequences.append(seq)
    
    return sequences


def _split_segments(text: str) -> List[str]:
    """Split text by --- delimiter."""
    return [seg.strip() for seg in text.split("---") if seg.strip()]
