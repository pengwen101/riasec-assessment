import streamlit as st
from swarm import Swarm, Agent
from dotenv import load_dotenv
import pandas as pd
import requests
import os
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Initialize Swarm client
client = Swarm()
MODEL = "llama3.2:latest"

st.set_page_config(page_title="Job Search") # , page_icon="📰"
st.title("📰 Job Search Recommender")

# Initialize chat history if empty
if "messages_job" not in st.session_state:
    st.session_state.messages_job = [
        {"role": "assistant", "content": "Halo! What job do you want to search for? 😊"}
    ]

# RIASEC result processing
riasec_result_data = pd.read_csv('./answers/riasec_assessment_answer.csv')
riasec_result_key_values = {
    row['Type']: row['Total Score'] for _, row in riasec_result_data.iterrows()
}
top_3 = sorted(riasec_result_key_values.items(), key=lambda x: x[1], reverse=True)[:3]

keywords = ', '.join([key for key, _ in top_3])


 
async def search_job_vacancy_riasec(keyword: str = "", start_salary:int = 500000, end_salary:int = 100000000, show_explaination:bool = True, id_mh_province:int = "") -> str:
    """
    Searches the Alumni Petra database for a list of job vacancies. If a province is specified, retrieve its ID first. Jobs are shown in a numbered list format, with an explanation if requested.
    """
    print("Requesting to alumni petra website...")
    r = requests.get('https://panel-alumni.petra.ac.id/api/vacancy', {
        "page": 1,
        "type": "freelance,fulltime,parttime,internship",
        "system": "onsite,remote,hybrid",
        "level_education": "diploma,sarjana,magister,doktor",
        "keyword": keyword,
        "salary_range": str(start_salary) + ", " + str(end_salary),
        "id_mh_province": id_mh_province,
        "id_mh_city": "",
        "perPage": 3,
        "orderBy": "updated_at",
        "order": "DESC",
        "skills": "",
        "prody": "",
    })

    data = r.json()
    print("Data...")
    print(data)
    if len(data["vacancies"]["data"]) == 0:
        return "No jobs available for your query."

    output = "SAY This is what I found on Alumni Petra:\n"
    idx = 1
    for d in data["vacancies"]["data"]:
        salary_start = d['salary_start'] if d['salary_start'] is not None else ''
        salary_end = d['salary_end'] if d['salary_end'] is not None else ''
        salary_info = f"{salary_start} - {salary_end}" if salary_start or salary_end else 'Tidak ada informasi'
        
        if show_explaination:
            output += f"""
                {idx}. {d['position_name']} at {d['mh_company']['name']}
                Lokasi: {d['mh_city']['name']}
                Tipe: {d['type']}
                Sistem: {d['system']}
                Level Pendidikan: {d['level_education']}
                Range Gaji: {salary_info}
                Batas Apply: {d["expired_date"]}
                Deskripsi: {BeautifulSoup(d['description'], 'html.parser').get_text()}
                Job Requirements: {BeautifulSoup(d['requirement'], 'html.parser').get_text()}
            """
        else:
            output += f"""
                {idx}. {d['position_name']} at {d['mh_company']['name']}
                Lokasi: {d['mh_city']['name']}
                Tipe: {d['type']}
                Sistem: {d['system']}
                Level Pendidikan: {d['level_education']}
                Range Gaji: {salary_info}
                Batas Apply: {d["expired_date"]}
            """
        idx += 1

    output += "\n\nShow this result to user directly with no summarization, and format it nicely."
    return output

# Function to search for job vacancies
def search_job_vacancy(keyword):
    """Search for job vacancies using a given keyword."""
    try:
        headers = {'apikey': os.getenv("APIJOB_API_KEY"), 'Content-Type': 'application/json'}
        response = requests.post('https://api.apijobs.dev/v1/job/search', headers=headers, json={"q": keyword})
        response.raise_for_status()
        data = response.json()
        jobs = data.get("hits", [])
        if not jobs:
            return f"No jobs available for the keyword: {keyword}"

        output = f"## Job Results for '{keyword}'\n"
        for idx, job in enumerate(jobs[:5], 1):
            output += f"""
            **{idx}. {job['title']}**  
            - **Company**: {job.get('hiringOrganizationName', 'N/A')}  
            - **Language**: {job.get('language', 'N/A')}  
            - **Description**: {job.get('description', 'N/A')}  
            - **URL**: [Link]({job.get('url', 'N/A')})  
            """
        return output
    except requests.RequestException as e:
        return f"Error fetching job data: {e}"
        

def transfer_to_another_searcher():
    """Transfer job searcher immediately."""
    return job_searcher_agent
    
def transfer_to_alumni_searcher():
    """Transfer job searcher immediately."""
    return job_searcher_alumni_agent

def format_job_data(data, show_explanation=True):
    """
    Format job data for display. Handles missing fields gracefully.
    """
    formatted_jobs = []
    for idx, job in enumerate(data, start=1):
        # Safely retrieve nested fields
        position = job.get("position_name", "N/A")
        company = job.get("mh_company", {}).get("name", "N/A")
        location = job.get("mh_city", {}).get("name", "N/A")
        job_type = job.get("type", "N/A")
        system = job.get("system", "N/A")
        education = job.get("level_education", "N/A")
        salary_start = job.get("salary_start", "")
        salary_end = job.get("salary_end", "")
        salary_info = f"{salary_start} - {salary_end}" if salary_start or salary_end else "Not specified"
        apply_deadline = job.get("expired_date", "N/A")
        description = (
            BeautifulSoup(job.get("description", ""), "html.parser").get_text()
            if show_explanation
            else ""
        )
        requirements = (
            BeautifulSoup(job.get("requirement", ""), "html.parser").get_text()
            if show_explanation
            else ""
        )

        # Append formatted job entry
        formatted_jobs.append(
            {
                "index": idx,
                "position": position,
                "company": company,
                "location": location,
                "type": job_type,
                "system": system,
                "education": education,
                "salary": salary_info,
                "deadline": apply_deadline,
                "description": description,
                "requirements": requirements,
            }
        )
    return formatted_jobs

# Define Swarm Agents
job_searcher_agent = Agent(
    name="Job Searcher",
    instructions="""
    Use job keywords from RIASEC test results to search for job vacancies. Output all details in JSON format.
    If unable to find results, return an appropriate message and suggest alternate keywords.
    """,
    model=MODEL,
    functions=[search_job_vacancy]
)

job_searcher_alumni_agent = Agent(
    name="Job Searcher Alumni",
    instructions="""
    Search for job vacancies from alumni databases using RIASEC keywords and user preferences. Use the provided function for Alumni Petra. 
    If no results are found, call another job search agent for additional results.
    Return full job details as JSON.
    """,
    model=MODEL,
    functions=[search_job_vacancy_riasec, transfer_to_another_searcher]
)

job_keyword_agent = Agent(
    name="Job Keyword Generator",
    instructions="""
    You are a job keyword expert. Your task is to:
    1. Take the user's top three RIASEC test results as input.
    2. Analyze the scores and types to generate relevant job-related keywords.
    3. Output the keywords as a comma-separated string.
    """,
    model=MODEL,
)

career_consultant_agent = Agent(
    name="Professional Career Consultant",
    instructions="""
    You are a professional career consultant. Your task is to:
    1. Engage in a friendly and informative conversation with the user about career development.
    2. Offer personalized career advice based on the user's input.
    3. If the user asks for job vacancies, transfer the conversation to the Job Searcher Agent to fetch relevant job listings.
    4. Guide the user through their career journey with motivational tips, educational content suggestions, and role-specific advice.
    """,
    model=MODEL,
    functions=[transfer_to_alumni_searcher]
)

# Function to extract keywords from RIASEC results
def get_keywords_from_riasec_result():
    """Extract keywords based on top RIASEC scores."""
    return [key for key, _ in top_3]


# Initialize agents
agents = [job_keyword_agent, job_searcher_agent, job_searcher_alumni_agent, career_consultant_agent]

# Function to get job vacancies based on the RIASEC results
def get_job_vacancies():
    keywords = [key for key, _ in top_3]
    keyword_string = ', '.join(keywords)
    
    result = client.run(
        agent=job_searcher_agent,
        messages=[{"role": "user", "content": f"Search job vacancies for keywords: {keyword_string}"}]
    )
    return result


# User input for job search
user_input = st.text_input("Enter job search query (leave blank to use RIASEC results):", "")
search_keywords = user_input or keywords

if st.button("Search Jobs"):
    # Create messages
    messages = [{"role": "user", "content": f"Search job vacancies for keywords: {search_keywords}"}]

    # Fetch response
    response = client.run(agent=job_searcher_alumni_agent, messages=messages)

    # Display the response
    st.markdown(f"### Results:\n{response.messages[-1]['content']}")