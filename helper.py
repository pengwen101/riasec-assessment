from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from bs4 import BeautifulSoup
import aiohttp

async def fetch_data(session: aiohttp.ClientSession , url: str) -> list:
    print(f"Starting task: {url}")
    async with session.get(url) as response:
        data = await response.json()
        jobs = []
        for vacancy in data['vacancies']['data']:
            raw_desc = vacancy.get('description') or 'No description provided'
            soup = BeautifulSoup(raw_desc, "html.parser")
            clean_desc = soup.get_text(separator=" ", strip=True)
            raw_req = vacancy.get('requirement') or 'No requirements provided'
            soup = BeautifulSoup(raw_req, "html.parser")
            clean_req = soup.get_text(separator=" ", strip=True)
            combined_text = f"DESCRIPTION:\n{clean_desc}\n\nREQUIREMENTS:\n{clean_req}"
            jobs.append({
                "slug": vacancy['slug'],
                "position_name": vacancy['position_name'],
                "job_context": combined_text
            })
        print(f"Fetched {len(jobs)} jobs from {url}")

        return jobs

async def sort_relevant_jobs(user_scores: dict, job_ratings: list) -> list:
    """
    Sorts jobs by Cosine Similarity.
    Returns the top 3 raw job dictionaries from Agent 1.
    """
    keys_ordered = list(user_scores.keys())
    user_vector = [user_scores[key] for key in keys_ordered]

    job_matrix = []
    for job_rating in job_ratings:
        job_vector = []
        for key in keys_ordered:
            job_vector.append(job_rating[key.lower()]["score"])
        job_matrix.append(job_vector)

    similarities = cosine_similarity([user_vector], job_matrix).flatten()
    sorted_similarities = np.argsort(similarities)[::-1]
    sorted_job_ratings = [job_ratings[idx] for idx in sorted_similarities]

    top_3_job_ratings = []
    for job_rating in sorted_job_ratings:
        # make sure the cached slug is still active
        url = f"https://panel-alumni.petra.ac.id/api/vacancy/{job_rating['job_slug']}"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            vacancy = data.get('vacancy', {})
            if vacancy.get('is_active') == 1:
                raw_desc = vacancy.get('description') or 'Tidak ada deskripsi'
                soup_desc = BeautifulSoup(raw_desc, "html.parser")
                clean_desc = list(soup_desc.stripped_strings)

                raw_req = vacancy.get('requirement') or 'Tidak ada persyaratan'
                soup_req = BeautifulSoup(raw_req, "html.parser")
                clean_req = list(soup_req.stripped_strings)

                mh_city = vacancy.get('mh_city') or {}
                mh_province = mh_city.get('mh_province') or {}
                mh_company = vacancy.get('mh_company') or {}

                job_rating.update({
                    "desc": clean_desc if clean_desc else ['Tidak ada deskripsi'],
                    "req": clean_req if clean_req else ['Tidak ada persyaratan'],
                    "expired_date": vacancy.get('expired_date', 'Tidak diketahui'),
                    "city": mh_city.get('name', 'Tidak diketahui'),
                    "province": mh_province.get('name', 'Tidak diketahui'),
                    "salary_start": vacancy.get('salary_start', 'N/A'),
                    "salary_end": vacancy.get('salary_end', 'N/A'),
                    "company": {
                        "name": mh_company.get('name', 'Perusahaan Tidak Diketahui'),
                        "business_scope": mh_company.get('bussiness_scope', 'Tidak diketahui') 
                    }
                })

                top_3_job_ratings.append(job_rating)
                if len(top_3_job_ratings) == 3:
                    break

    return top_3_job_ratings


async def calculate_gaps(user_scores: dict, top_3_job_ratings: list) -> list:
    """
    Finds gaps where the job requires a score >= 5
    and the difference between job's score and user's score is significant (>=1).
    Returns a list of dictionaries perfectly matching the JobKeywordGaps schema.
    """
    traits = list(user_scores.keys())
    normalized_user_scores = {trait: ((user_scores[trait]-8) / 32.0) * 10 for trait in traits}

    result = []

    for job_rating in top_3_job_ratings:
        job_payload = {
            "job_slug": job_rating["job_slug"],
            "job_title": job_rating["job_title"],
            "gaps": {}
        }

        for trait in traits:
            trait_lower = trait.lower()
            job_req_score = job_rating[trait_lower]["score"]
            user_score_val = normalized_user_scores[trait]

            if job_req_score >= 5 and job_req_score - user_score_val >= 1.0:
                job_payload["gaps"][trait_lower] = {
                    "trait_tendency": job_rating[trait_lower]["reasoning"]["trait_tendency"],
                    "job": {
                        "score": job_req_score,
                        "job_context_quote": job_rating[trait_lower]["reasoning"]["job_context_quote"],
                        "logic": job_rating[trait_lower]["reasoning"]["logic"]
                    },
                    "user_score": int(user_score_val),
                    "keywords": []
                }

        if job_payload["gaps"]:
            result.append(job_payload)

    return result