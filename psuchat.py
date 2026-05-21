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

    # Choose focused PSU pages based on the question
    if "housing" in question_lower or "dining" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/housing-and-food-services",
            "https://foodservices.psu.edu/",
            "https://liveon.psu.edu/"
        ]

    elif "program" in question_lower or "major" in question_lower or "academic" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/academic-programs",
            "https://bulletins.psu.edu/programs/",
            "https://harrisburg.psu.edu/academics"
        ]

    elif "admission" in question_lower or "apply" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/admissions",
            "https://admissions.psu.edu/"
        ]

    elif "financial" in question_lower or "aid" in question_lower or "scholarship" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/financial-aid",
            "https://studentaid.psu.edu/"
        ]

    elif "canvas" in question_lower:

        urls = [
            "https://canvas.psu.edu/"
        ]

    elif "lionpath" in question_lower:

        urls = [
            "https://lionpath.psu.edu/"
        ]

    elif "career" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/career-services"
        ]

    elif "student" in question_lower or "campus" in question_lower:

        urls = [
            "https://harrisburg.psu.edu/student-life",
            "https://harrisburg.psu.edu/campus-life"
        ]

    else:

        urls = [
            "https://harrisburg.psu.edu/",
            "https://harrisburg.psu.edu/academics",
            "https://harrisburg.psu.edu/admissions",
            "https://harrisburg.psu.edu/student-life"
        ]

    all_text = ""

    for url in urls:

        try:

            response = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(" ", strip=True)

            cleaned_text = text[:12000]

            all_text += f"""

SOURCE WEBSITE:
{url}

WEBSITE CONTENT:
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

                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful Penn State Harrisburg assistant.

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

Only reject questions that are clearly unrelated like:
- sports scores
- celebrities
- politics
- weather
- random programming questions

IMPORTANT:
- Use ONLY the website information provided below.
- Summarize the information clearly.
- Use bullet points.
- Do NOT say:
    - "I do not have access"
    - "I cannot provide"
    - "I do not know"
- If information is limited, still summarize what is available.
- Always include the most relevant source website URL at the end.

WEBSITE INFORMATION:
{website_info}

USER QUESTION:
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