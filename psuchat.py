import os
import streamlit as st
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="PSU Harrisburg Assistant")
st.title("PSU Harrisburg Website Assistant")


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


def scrape_psu_pages(question):
    q = question.lower()

    if "housing" in q or "dining" in q or "meal" in q:
        urls = [
            "https://liveon.psu.edu/harrisburg",
            "https://liveon.psu.edu/meal-plans",
            "https://foodservices.psu.edu/"
        ]

    elif "major" in q or "program" in q or "academic" in q:
        urls = [
            "https://harrisburg.psu.edu/academics",
            "https://harrisburg.psu.edu/academic-programs",
            "https://bulletins.psu.edu/programs/"
        ]

    elif "admission" in q or "apply" in q:
        urls = [
            "https://harrisburg.psu.edu/admissions",
            "https://admissions.psu.edu/"
        ]

    elif "financial" in q or "aid" in q or "tuition" in q or "scholarship" in q:
        urls = [
            "https://harrisburg.psu.edu/tuition-and-financial-aid",
            "https://studentaid.psu.edu/",
            "https://tuition.psu.edu/"
        ]

    elif "canvas" in q:
        urls = ["https://canvas.psu.edu/"]

    elif "lionpath" in q:
        urls = ["https://lionpath.psu.edu/"]

    elif "career" in q:
        urls = ["https://harrisburg.psu.edu/career-services"]

    elif "registrar" in q or "calendar" in q:
        urls = [
            "https://harrisburg.psu.edu/registrar",
            "https://www.registrar.psu.edu/academic-calendars/"
        ]

    else:
        urls = [
            "https://harrisburg.psu.edu/",
            "https://harrisburg.psu.edu/academics",
            "https://harrisburg.psu.edu/admissions",
            "https://harrisburg.psu.edu/student-life",
            "https://harrisburg.psu.edu/campus-life"
        ]

    all_text = ""

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "noscript", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text(" ", strip=True)

            all_text += f"\n\nSOURCE WEBSITE: {url}\n{text[:12000]}"

        except Exception as e:
            all_text += f"\n\nSOURCE WEBSITE: {url}\nERROR: {e}"

    return all_text


question = st.chat_input("Ask something about PSU Harrisburg")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PSU Harrisburg websites..."):

            api_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")

            if not api_key:
                st.error("Missing OPENROUTER_API_KEY.")
            else:
                website_info = scrape_psu_pages(question)

                with st.expander("View scraped website text"):
                    st.write(website_info[:10000])

                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful Penn State Harrisburg assistant.

Questions about PSU Harrisburg academics, housing, dining, admissions, financial aid,
campus life, majors, programs, Canvas, LionPATH, registrar, and student services
are related to Penn State Harrisburg.

Only reject clearly unrelated questions.

Use ONLY the website text below.

Rules:
- Answer in simple bullet points.
- Include the source website URL.
- Do not make up facts.
- If the website text has partial information, summarize what is available.

WEBSITE TEXT:
{website_info}

QUESTION:
{question}
"""

                try:
                    response = llm.invoke([HumanMessage(content=prompt)])
                    st.write(response.content)

                except Exception as e:
                    st.error("OpenRouter error.")
                    st.write(e)