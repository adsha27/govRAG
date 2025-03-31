import requests
import json

class MCQGenerator:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"

    def generate_mcq(self, chunk: str) -> str:
        prompt = f"""
You are an expert teacher creating high-quality exam-style multiple-choice questions from educational text.

Text:
\"\"\"
{chunk.strip()}
\"\"\"

Instructions:
- Generate exactly ONE question that tests **foundational understanding**.
- Keep the **question concise**, no more than 4–5 sentences.
- Each option (A–D) should be 1–2 sentences maximum.
- Ensure the answer requires reasoning, not direct recall.
- End with: Correct Answer: <A/B/C/D>

Output format:
Question: ...
A) ...
B) ...
C) ...
D) ...
Correct Answer: ...
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip() or "[Error: Empty response]"
        except Exception as e:
            print(f"[Ollama API Error] {e}")
            return "[Error generating MCQ]"

if __name__ == "__main__":
    sample_chunk = (
        "Newton’s Second Law states that the acceleration of an object is directly proportional to the net force "
        "acting upon it and inversely proportional to its mass. This principle can be used to analyze the motion "
        "of objects in a variety of mechanical systems."
    )

    generator = MCQGenerator(model_name="llama3")
    mcq = generator.generate_mcq(sample_chunk)
    print("Generated MCQ:\n")
    print(mcq)