from pydantic import BaseModel
from typing import List, Dict

class Reasoning(BaseModel):
    job_context_quote: str
    trait_tendency: str
    logic: str

class ScoreReasonPair(BaseModel):
    score: int
    reasoning: Reasoning

class JobRating(BaseModel):
    job_slug: str
    job_title: str
    realistic: ScoreReasonPair
    investigative: ScoreReasonPair
    artistic: ScoreReasonPair
    social: ScoreReasonPair
    enterprising: ScoreReasonPair
    conventional: ScoreReasonPair

class BatchJobRatings(BaseModel):
    ratings: List[JobRating]

class JobDetails(BaseModel):
    score: int
    job_context_quote: str
    logic: str

class GapDetailOut(BaseModel):
    trait_tendency: str
    job: JobDetails
    user_score: int
    keywords: List[str]

class JobKeywordGaps(BaseModel):
    job_slug: str
    job_title: str
    gaps: Dict[str, GapDetailOut]

class BatchKeywordsOut(BaseModel):
    data: List[JobKeywordGaps]