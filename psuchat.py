import os
import streamlit as st

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
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


firecrawl = FirecrawlApp(
    api_key=get_secret("FIRECRAWL_API_KEY")
)


def get_start_url(question):
    q = question.lower()

    if "housing" in q or "dorm" in q or "room" in q:
        return "https://liveon.psu.edu/harrisburg"

    elif "meal" in q or "dining" in q or "food" in q:
        return "https://liveon.psu.edu/meal-plans"

    elif "major" in q or "program" in q or "academic" in q:
        return "https://harrisburg.psu.edu/academics"

    elif "admission" in q or "apply" in q:
        return "https://harrisburg.psu.edu/admissions"

    elif "financial" in q or "aid" in q or "scholarship" in q:
        return "https://harrisburg.psu.edu/financial-aid"

    elif "tuition" in q or "cost" in q:
        return "https://tuition.psu.edu/"

    else:
        return "https://harrisburg.psu.edu/"


def scrape_with_firecrawl(question):
    start_url = get_start_url(question)

    try:
        result = firecrawl.scrape_url(
            start_url,
            formats=["markdown"]
        )

        markdown = result.get("markdown", "")

        return f"""
SOURCE WEBSITE:
{start_url}

SCRAPED WEBSITE TEXT:
{markdown}
"""

    except Exception as e:
        return f"""
SOURCE WEBSITE:
{start_url}

ERROR:
{e}
"""


question = st.chat_input("Ask something about PSU Harrisburg")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PSU website with Firecrawl..."):

            api_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")
            firecrawl_key = get_secret("FIRECRAWL_API_KEY")

            if not api_key:
                st.error("Missing OPENROUTER_API_KEY.")

            elif not firecrawl_key:
                st.error("Missing FIRECRAWL_API_KEY.")

            else:
                website_info = scrape_with_firecrawl(question)

                with st.expander("View website text"):
                    st.write(website_info[:12000])

                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful PSU Harrisburg assistant.

Use ONLY the website text below.

Rules:
- Answer in simple bullet points.
- Include the source website URL.
- If the website has partial information, still answer with what is available.
- Do not make up facts.

WEBSITE TEXT:
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