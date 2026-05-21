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

    question_lower = question.lower()

    # HOUSING / DINING
    if "housing" in question_lower or "dining" in question_lower or "meal" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/housing",
            "https://liveon.psu.edu/harrisburg",
            "https://liveon.psu.edu/harrisburg/housing-options",
            "https://liveon.psu.edu/harrisburg/rates",
            "https://liveon.psu.edu/meal-plans",
            "https://foodservices.psu.edu/",
            "https://harrisburg.psu.edu/residence-life"
        ]

    # ACADEMICS / MAJORS
    elif "program" in question_lower or "major" in question_lower or "academic" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/academics",
            "https://harrisburg.psu.edu/academic-programs",
            "https://bulletins.psu.edu/programs/"
        ]

    # ADMISSIONS
    elif "admission" in question_lower or "apply" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/admissions",
            "https://admissions.psu.edu/"
        ]

    # FINANCIAL AID
    elif "financial" in question_lower or "aid" in question_lower or "scholarship" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/financial-aid",
            "https://studentaid.psu.edu/"
        ]

    # CAMPUS LIFE
    elif "campus" in question_lower or "student" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/campus-life",
            "https://harrisburg.psu.edu/student-life"
        ]

    # CANVAS
    elif "canvas" in question_lower:

        urls = [
            "https://canvas.psu.edu/"
        ]

    # LIONPATH
    elif "lionpath" in question_lower:

        urls = [
            "https://lionpath.psu.edu/"
        ]

    # CAREER SERVICES
    elif "career" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/career-services"
        ]

    # DEFAULT
    else:

        urls = [
            "https://harrisburg.psu.edu/",
            "https://harrisburg.psu.edu/academics",
            "https://harrisburg.psu.edu/admissions",
            "https://harrisburg.psu.edu/student-life"
        ]

    all_text = ""

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    for url in urls:

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "noscript"]):
                tag.extract()

            text = soup.get_text(separator=" ", strip=True)

            cleaned_text = text[:15000]

            all_text += f"""

SOURCE WEBSITE:
{url}

SCRAPED WEBSITE TEXT:
{cleaned_text}

"""

        except Exception as e:

            all_text += f"""

SOURCE WEBSITE:
{url}

ERROR:
{e}

"""

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
You are a helpful PSU Harrisburg assistant.

Questions about:
- academics
- housing
- dining
- admissions
- financial aid
- campus life
- majors
- programs
- Canvas
- LionPATH
- PSU student services

ARE related to Penn State Harrisburg.

Only reject clearly unrelated questions.

IMPORTANT:
- Use ONLY the scraped website text below.
- Summarize the information.
- Use bullet points.
- NEVER say:
    - "I do not have access"
    - "I cannot provide"
    - "information is unavailable"

If partial information exists, summarize it anyway.

Always include the source website URL.

SCRAPED WEBSITE TEXT:
{website_info}

QUESTION:
{question}
"""

                try:

                    response = llm.invoke([
                        HumanMessage(content=prompt)
                    ])

                    st.write(response.content)

                except Exception as e:

                    st.error("OpenRouter error.")
                    st.write(e)