import json
from llama_index.llms.ollama import Ollama
from schemas import BatchKeywordsOut

def extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM output:\n{text[:300]}")

    return json.loads(text[start:end])

async def generate_keyword_for_batch(batch_gaps: list, llm: Ollama, batch_num: int) -> list:
    """
    Asks the LLM ONLY for the keywords.
    """

    for job in batch_gaps:
        for trait, gap_info in job["gaps"].items():
            prompt = f"""You are an Educational Content Curator.
A user has a skill gap in the "{trait}" trait for the role of {job['job_title']}.

CONTEXT FROM CAREER ADVISOR:
Job Context Quote: "{gap_info['job'].get('job_context_quote', 'N/A')}"
Logic: "{gap_info['job'].get('logic', 'N/A')}"

TASK:
Generate exactly 3 highly specific, actionable YouTube search keywords to help the user learn this missing skill.

OUTPUT FORMAT:
Return ONLY a flat JSON object with a single "keywords" array. Do not write anything else.
{{
  "keywords": [
    "keyword 1",
    "keyword 2",
    "keyword 3"
  ]
}}"""

            response = await llm.acomplete(prompt)
            raw_text = response.text if hasattr(response, "text") else str(response)
            try:
                parsed = extract_json(raw_text)
                gap_info["keywords"] = parsed.get("keywords", [])
                print(f"[Batch {batch_num}] Generated keywords for {job['job_title']} - {trait}")
            except Exception as e:
                print(f"[Batch {batch_num}] Error parsing keywords for {trait}: {e}")
                gap_info["keywords"] = ["Error generating keywords"]
    try:
        validated = BatchKeywordsOut(data=batch_gaps)
        return validated.model_dump()["data"]
    except Exception as e:
        print(f"[Batch {batch_num}] CRITICAL Pydantic Error after injection: {e}")
        raise e

async def run_keyword_pipeline(gaps: list, batch_size: int = 5):
    llm = Ollama(model="gemma4:latest", request_timeout=180.0)
    all_results = []

    for i in range(0, len(gaps), batch_size):
        batch = gaps[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"\n── Processing batch {batch_num} ({len(batch)} jobs) ──")
        try:
            result_list = await generate_keyword_for_batch(batch, llm, batch_num)
            all_results.extend(result_list)
        except Exception as e:
            print(f"[Batch {batch_num}] ERROR: {e}")

    return all_results