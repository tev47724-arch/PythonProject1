import os
import streamlit as st
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.title("PSU Harrisburg Website Assistant")


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


def scrape_psu_pages(question):

    # choose ONE focused website based on question
    if "housing" in question.lower() or "dining" in question.lower():
        urls = [
            "https://harrisburg.psu.edu/housing-and-food-services",
            "https://foodservices.psu.edu/",
            "https://liveon.psu.edu/"
        ]

    elif "admission" in question.lower():
        urls = [
            "https://harrisburg.psu.edu/admissions",
            "https://admissions.psu.edu/"
        ]

    elif "financial" in question.lower() or "aid" in question.lower():
        urls = [
            "https://harrisburg.psu.edu/financial-aid",
            "https://studentaid.psu.edu/"
        ]

    elif "canvas" in question.lower():
        urls = [
            "https://canvas.psu.edu/"
        ]

    elif "lionpath" in question.lower():
        urls = [
            "https://lionpath.psu.edu/"
        ]

    elif "program" in question.lower() or "major" in question.lower():
        urls = [
            "https://harrisburg.psu.edu/academic-programs",
            "https://bulletins.psu.edu/programs/"
        ]

    else:
        urls = [
            "https://harrisburg.psu.edu/",
            "https://harrisburg.psu.edu/student-life",
            "https://harrisburg.psu.edu/campus-life"
        ]

    all_text = ""

    for url in urls:

        try:
            response = requests.get(url, timeout=10)

            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(" ", strip=True)

            all_text += f"\n\nSOURCE WEBSITE: {url}\n{text[:12000]}"

        except Exception as e:

            all_text += f"\nCould not read {url}: {e}"

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
You are a PSU Harrisburg assistant.

ONLY answer questions related to Penn State Harrisburg.

If unrelated, say:
"I can only answer questions related to Penn State Harrisburg."

Answer using ONLY the website information below.

Do not say:
- I do not have access
- I cannot provide
- I do not know

Instead summarize the information you find.

Use bullet points.

At the end include the source website.

Website Information:
{website_info}

Question:
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