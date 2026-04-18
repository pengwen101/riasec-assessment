import streamlit as st
import pandas as pd
import asyncio
import aiohttp
from agents.search_and_rate_agent import run_job_rating_pipeline
from agents.generate_video_keywords import run_keyword_pipeline
from helper import sort_relevant_jobs, calculate_gaps, fetch_data
import logging
import sys
import urllib.parse

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


async def run_full_analysis(user_scores):
    urls = [
        f"https://panel-alumni.petra.ac.id/api/vacancy?page={page}&type=freelance,fulltime,parttime,internship&system=onsite,remote,hybrid&level_education=diploma,sarjana,magister,doktor&keyword=&salary_range=0,100000000&id_mh_province=&id_mh_city=&perPage=10&orderBy=updated_at&order=DESC&skills=&prody="
        for page in range(1, 5)
    ]

    st.info("🌐 Fetching latest job vacancies...")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url) for url in urls]
        pages = await asyncio.gather(*tasks)

    jobs = [job for page in pages for job in page] # rich jobs

    st.info(f"🤖 Agent 1: Rating jobs against the RIASEC framework...")
    job_ratings = await run_job_rating_pipeline(
        jobs=jobs,
        document_path="./docs/career-theory-model-holland-20170501.pdf",
        cache_path=".cache/job_rating_agent_result.json",
        batch_size=2
    )

    st.info("🔍 Searching top 3 jobs matching your profile...")
    top_3_job_ratings = await sort_relevant_jobs(user_scores, job_ratings)
    print(top_3_job_ratings)

    st.info("🧮 Calculating skill gaps between your profile and job requirements...")
    gap_data = await calculate_gaps(user_scores, top_3_job_ratings)

    if gap_data:
        st.info("🎓 Agent 2: Generating educational YouTube keywords for your gaps...")
        keyword_results = await run_keyword_pipeline(gaps=gap_data, batch_size=1)
        for enriched_job in top_3_job_ratings:
            current_slug = enriched_job.get('job_slug')
            matching_gap = next((k for k in keyword_results if k.get('job_slug') == current_slug), None)
            if matching_gap and 'gaps' in matching_gap:
                enriched_job['gaps'] = matching_gap['gaps']
    else:
        st.info("🎓 No significant skill gaps found!")

    return top_3_job_ratings

if 'page' not in st.session_state:
    st.session_state.page = "quiz"

if 'user_scores' not in st.session_state:
    st.session_state.user_scores = {}


if st.session_state.page == "quiz":
    st.title("RIASEC Career Assessment")

    tab1, tab2 = st.tabs(["📝 Isi Kuesioner", "⚡ Input Skor Langsung"])

    with tab1:
        try:
            questions_df = pd.read_csv("./questions/holland-questions.csv")
        except FileNotFoundError:
            st.error("Could not find the questions CSV. Please check the path.")
            st.stop()

        with st.form("riasec_form"):
            st.write("Jawablah pertanyaan berikut dengan skala 1-5:")
            st.write("**(1 = Sangat Tidak Yakin, 2 = Tidak Yakin, 3 = Netral, 4 = Yakin, 5 = Sangat Yakin)**")

            if 'answers' not in st.session_state:
                st.session_state.answers = {}

            st.subheader("Questions:")

            for index, row in questions_df.iterrows():
                question = row['Question']
                st.session_state.answers[f"answer_{index}"] = st.slider(
                    f"{index+1}. {question}", 
                    min_value=1, max_value=5, value=1, format="%d"
                )

            st.markdown("""
                <style>
                div.stButton > button { display: block; margin: 0 auto; }
                </style>
                """, unsafe_allow_html=True)

            submitted_quiz = st.form_submit_button("Submit Assessment")

        if submitted_quiz:
            total_scores = {type_: 0 for type_ in questions_df['Type'].unique()}
            for index in range(len(questions_df)):
                type_ = questions_df.loc[index, 'Type']
                score = st.session_state.answers[f'answer_{index}']
                total_scores[type_] += score

            st.session_state.user_scores = total_scores

            total_scores_df = pd.DataFrame(total_scores.items(), columns=['Type', 'Total Score'])
            st.session_state.top_3 = total_scores_df.sort_values(by='Total Score', ascending=False).head(3)

            st.session_state.quiz_completed = True 

    with tab2:
        st.write("Sudah mengetahui persentase skor RIASEC Anda? Sesuaikan slider di bawah ini (0% - 100%):")

        with st.form("direct_input_form"):
            col1, col2 = st.columns(2)

            with col1:
                r_pct = st.slider("Realistic (R)", min_value=0, max_value=100, value=50, format="%d%%")
                i_pct = st.slider("Investigative (I)", min_value=0, max_value=100, value=50, format="%d%%")
                a_pct = st.slider("Artistic (A)", min_value=0, max_value=100, value=50, format="%d%%")

            with col2:
                s_pct = st.slider("Social (S)", min_value=0, max_value=100, value=50, format="%d%%")
                e_pct = st.slider("Enterprising (E)", min_value=0, max_value=100, value=50, format="%d%%")
                c_pct = st.slider("Conventional (C)", min_value=0, max_value=100, value=50, format="%d%%")

            submitted_direct = st.form_submit_button("Submit Skor Langsung")

        if submitted_direct:
            def convert_to_internal_score(pct):
                return int(round(8 + (pct / 100.0) * 32))

            total_scores = {
                "Realistic": convert_to_internal_score(r_pct),
                "Investigative": convert_to_internal_score(i_pct),
                "Artistic": convert_to_internal_score(a_pct),
                "Social": convert_to_internal_score(s_pct),
                "Enterprising": convert_to_internal_score(e_pct),
                "Conventional": convert_to_internal_score(c_pct)
            }

            st.session_state.user_scores = total_scores

            total_scores_df = pd.DataFrame(total_scores.items(), columns=['Type', 'Total Score'])
            st.session_state.top_3 = total_scores_df.sort_values(by='Total Score', ascending=False).head(3)

            st.session_state.quiz_completed = True


    if st.session_state.get('quiz_completed', False):
        st.success("Jawaban anda telah berhasil disimpan!")

        st.markdown(
            f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; background-color: #f9f9f9; margin-bottom: 20px;">
                <h3 style="color: #4CAF50;">Top 3 Trait Anda:</h3>
                <h5>1. {st.session_state.top_3.iloc[0, 0]} ({st.session_state.top_3.iloc[0, 1]} pts)</h5>
                <h5>2. {st.session_state.top_3.iloc[1, 0]} ({st.session_state.top_3.iloc[1, 1]} pts)</h5>
                <h5>3. {st.session_state.top_3.iloc[2, 0]} ({st.session_state.top_3.iloc[2, 1]} pts)</h5>
            </div>
            """, unsafe_allow_html=True
        )

        if st.button("Analisa Lowongan Pekerjaan & Cari Gap Pembelajaran 🚀"):
            st.session_state.quiz_completed = False 
            st.session_state.page = "agents"
            st.rerun()


elif st.session_state.page == "agents":
    st.title("AI Career Gap Analysis")
    st.write("Menganalisa profil RIASEC Anda dengan lowongan pekerjaan saat ini...")

    with st.spinner("Pipeline AI sedang berjalan. Mohon tunggu..."):
        try:
            top_3_job_ratings_detail = asyncio.run(run_full_analysis(st.session_state.user_scores))
            st.success("Analisis Selesai!")
            st.subheader("3 Pekerjaan Tercocok Dengan Profil RIASEC-mu!")

            def format_rupiah(amount):
                try:
                    return f"Rp {int(amount):,}".replace(',', '.')
                except (ValueError, TypeError):
                    return str(amount)

            for job in top_3_job_ratings_detail:
                job_title = job.get('job_title', 'Posisi Tidak Diketahui')
                job_slug = job.get('job_slug', '')
                full_job_url = f"https://alumni.petra.ac.id/vacancy/{job_slug}"
                company_data = job.get('company', {})
                company_name = company_data.get('name', 'Perusahaan Tidak Diketahui') if isinstance(company_data, dict) else str(company_data)
                company_scope = company_data.get('business_scope', 'Tidak diketahui') if isinstance(company_data, dict) else 'Tidak diketahui'

                sal_start = format_rupiah(job.get('salary_start', 0))
                sal_end = format_rupiah(job.get('salary_end', 0))

                with st.expander(f"📌 {job_title} at {company_name}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**🏢 {company_name}**")
                        st.caption(f"Scope Perusahaan: {company_scope}")
                        st.markdown(f"📍 **Lokasi:** {job.get('city', 'Kota tidak diketahui')}, {job.get('province', '')}")
                        st.markdown(f"💰 **Gaji:** {sal_start} - {sal_end}")
                        st.caption(f"Expired: {job.get('expired_date', 'Tidak diketahui')}")
                    with col2:
                        st.link_button("🌐 Buka di Web", full_job_url)

                    st.divider()

                    st.markdown("#### 📝 Deskripsi Pekerjaan")
                    for desc in job.get('desc', []):
                        st.markdown(f"- {desc}")
                    st.markdown("#### 🎯 Persyaratan")
                    for req in job.get('req', []):
                        st.markdown(f"- {req}")
                    st.divider()

                    st.markdown("### 📊 Analisis RIASEC: Pekerjaan vs Anda")

                    traits = ['realistic', 'investigative', 'artistic', 'social', 'enterprising', 'conventional']

                    for trait in traits:
                        trait_data = job.get(trait, {})
                        if not isinstance(trait_data, dict):
                            trait_data = {}

                        job_raw_score = trait_data.get("score", 0)
                        job_percentage = int((job_raw_score / 10.0) * 100)

                        user_raw_score = st.session_state.user_scores.get(trait.capitalize(), 8)
                        user_percentage = int(((user_raw_score - 8) / 32.0) * 100)

                        st.markdown(f"#### **{trait.capitalize()}**")

                        col_job, col_user = st.columns(2)
                        with col_job:
                            st.caption(f"🏢 Skor Pekerjaan: **{job_percentage}%**")
                            st.progress(job_percentage)
                        with col_user:
                            st.caption(f"👤 Skor Anda: **{user_percentage}%**")
                            st.progress(user_percentage)

                        reasoning = trait_data.get("reasoning")
                        trait_tendency = reasoning.get("trait_tendency")
                        quote = reasoning.get("job_context_quote")
                        logic = reasoning.get("logic")

                        if quote or logic:
                            st.info(
                                f"**💡 Mengapa pekerjaan ini membutuhkan {trait.capitalize()} sebesar {job_percentage}%?**\n\n"
                                f"> *\"{quote}\"*\n\n"
                                f"{trait_tendency}\n"
                                f"{logic}"
                            )

                        gaps = job.get('gaps', {})
                        if trait in gaps:
                            st.warning("⚠️ **Skill Gap Ditemukan!** Anda disarankan untuk mempelajari:")
                            keywords = gaps[trait].get("keywords", [])
                            for kw in keywords:
                                yt_query = urllib.parse.quote_plus(kw)
                                yt_url = f"https://www.youtube.com/results?search_query={yt_query}"
                                st.markdown(f"- 📺 [{kw}]({yt_url})")

                        st.markdown("<br>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan pada pipeline AI: {str(e)}")

    if st.button("← Kembali ke Quiz"):
        st.session_state.page = "quiz"
        st.rerun()