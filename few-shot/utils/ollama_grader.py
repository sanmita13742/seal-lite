"""
Ollama-based grading system to replace OpenAI GPT-4 grading.
"""
import requests
import re
from typing import List


class OllamaGrader:
    """Grade model outputs using local Ollama model."""
    
    def __init__(self, model: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.session = requests.Session()
        
    def grade_yes_no(self, prompt: str) -> bool:
        """
        Send grading prompt to Ollama and parse yes/no response.
        """
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 10,  # Short response needed
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "").lower().strip()
                return self._parse_yes_no(text)
            else:
                print(f"Ollama grading error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error in Ollama grading: {e}")
            return False
    
    def grade_batch(self, prompts: List[str]) -> List[bool]:
        """Grade multiple prompts."""
        return [self.grade_yes_no(p) for p in prompts]
    
    @staticmethod
    def _parse_yes_no(text: str) -> bool:
        """Parse yes/no from response text."""
        yes_pattern = re.compile(r'\b(yes)\b', re.IGNORECASE)
        no_pattern = re.compile(r'\b(no)\b', re.IGNORECASE)
        
        # Check for yes without no
        if yes_pattern.search(text) and not no_pattern.search(text):
            return True
        return False


# Grading template (same as original)
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


def format_grade_prompts(questions: List[dict], predictions: List[str]) -> List[str]:
    """Format grading prompts."""
    return [
        SQUAD_GRADE_TEMPLATE.format(
            question=q["question"],
            gold=q["answer"],
            pred=p.strip()
        )
        for q, p in zip(questions, predictions)
    ]
