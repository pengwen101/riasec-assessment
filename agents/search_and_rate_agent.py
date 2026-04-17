import json

from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from schemas import BatchJobRatings

Settings.chunk_size = 384
Settings.chunk_overlap = 50

def build_index(document_path: str):
    documents = SimpleDirectoryReader(input_files=[document_path]).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=OllamaEmbedding(model_name="mxbai-embed-large")
    )
    return index

def extract_json(text: str) -> dict:
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

async def rate_batch(batch: list, riasec_context: str, llm: Ollama, batch_num: int) -> list:
    jobs_text = "\n".join(
        f"slug: {j['slug']}\ntitle: {j['position_name']}\njob_context: {j['job_context'][:400]}"
        for j in batch
    )

    example_output = """{
  "ratings": [
    {
      "job_slug": "example-slug",
      "job_title": "Example Job",
      "realistic": {
        "score": 8,
        "reasoning": {
          "job_context_quote": "Must lift 50 lbs and operate heavy machinery.",
          "trait_tendency": "Realistic individuals prefer working with tools, machines, and physical tasks.",
          "logic": "The job explicitly requires heavy physical labor and machinery operation, perfectly matching the Realistic trait."
        }
      },
      "investigative": {
        "score": 2,
        "reasoning": {
          "job_context_quote": "Follow standard operating procedures.",
          "trait_tendency": "Investigative individuals like analyzing and solving complex problems.",
          "logic": "The role relies on established procedures rather than deep analytical problem-solving."
        }
      },
      "artistic": {"score": 0, "reasoning": {"job_context_quote": "...", "trait_tendency": "...", "logic": "..."}},
      "social": {"score": 0, "reasoning": {"job_context_quote": "...", "trait_tendency": "...", "logic": "..."}},
      "enterprising": {"score": 0, "reasoning": {"job_context_quote": "...", "trait_tendency": "...", "logic": "..."}},
      "conventional": {"score": 0, "reasoning": {"job_context_quote": "...", "trait_tendency": "...", "logic": "..."}}
    }
  ]
}"""

    prompt = f"""You are an expert Career Advisor using the RIASEC personality framework.

RIASEC FRAMEWORK REFERENCE:
{riasec_context}

Rate each job on ALL SIX RIASEC dimensions using a 0-10 scale (10 = extremely relevant).

OUTPUT FORMAT — return ONLY a raw JSON object exactly like this example (no markdown, no explanation):
{example_output}

Each entry MUST contain these exact keys:
  job_slug, job_title, realistic, investigative, artistic, social, enterprising, conventional

STRICT SCORING RULES:
1. You must base your score ONLY on the exact text provided in the "job_context" (Description and Requirements).
2. DO NOT guess or assume responsibilities that are not written in the text.
3. If the provided text is empty, too short, or lacks any evidence to support a specific RIASEC trait, you MUST:
   - Set the "score" to 0.
   - Set the "job_context_quote" to "N/A".
   - Set the "logic" to "Insufficient data provided in the job description and requirements to evaluate this trait."

JOBS TO RATE:
{jobs_text}

Return ONLY the JSON object. Do not write anything before or after it."""

    response = llm.acomplete(prompt)
    raw_text = response.text if hasattr(response, "text") else str(response)
    print(f"[Batch {batch_num}] LLM response received ({len(raw_text)} chars).")

    parsed = extract_json(raw_text)
    validated = BatchJobRatings(**parsed)
    print(f"[Batch {batch_num}] Validated {len(validated.ratings)} ratings successfully.")

    return validated.model_dump()["ratings"]

async def run_job_rating_pipeline(jobs: list, document_path: str, cache_path: str, batch_size: int = 5):
    print("Building RIASEC vector index...")
    index = build_index(document_path)
    llm = Ollama(model="gemma4:latest", request_timeout=180.0, verbose=True, json_mode=True)
    print("Index ready.\n")

    try:
        with open(cache_path, 'r') as f:
            cached = json.load(f)
            print(f"Loaded {len(cached)} from cache.")
    except (FileNotFoundError, json.JSONDecodeError):
        cached = []

    all_results = list(cached)
    cached_slugs = {job['job_slug'] for job in all_results if 'job_slug' in job}
    jobs = [job for job in jobs if job['slug'] not in cached_slugs]

    if len(jobs) > 0:
        print("Generating RIASEC summary from document...")
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=4)
        riasec_context = str(query_engine.query(
            "Summarize the definitions, traits, and ideal work environments "
            "for all six RIASEC types: Realistic, Investigative, Artistic, "
            "Social, Enterprising, Conventional."
        ))
        print("Summary generated.\n")

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"\n── Processing batch {batch_num} ({len(batch)} jobs) ──")
        try:
            result_list = await rate_batch(batch, riasec_context, llm, batch_num)
            all_results.extend(result_list)
        except Exception as e:
            print(f"[Batch {batch_num}] ERROR: {e}")

    with open(cache_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results