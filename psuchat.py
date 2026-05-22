import os
import requests
import streamlit as st

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
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


def search_psu_pages(question, max_results=5):
    search_query = f"site:harrisburg.psu.edu {question}"

    urls = []

    with DDGS() as ddgs:
        results = ddgs.text(search_query, max_results=max_results)

        for result in results:
            url = result.get("href")

            if url and "harrisburg.psu.edu" in url:
                urls.append(url)

    return urls


def scrape_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(" ", strip=True)

        return f"""
SOURCE WEBSITE:
{url}

SCRAPED WEBSITE TEXT:
{text[:7000]}
"""

    except Exception as e:
        return f"""
SOURCE WEBSITE:
{url}

ERROR:
{e}
"""


def get_website_info(question):
    urls = search_psu_pages(question)

    if not urls:
        return "No PSU Harrisburg pages found."

    all_text = ""

    for url in urls:
        all_text += scrape_page(url)

    return all_text


question = st.chat_input("Ask something about PSU Harrisburg")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PSU Harrisburg website..."):

            api_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")

            if not api_key:
                st.error("Missing OPENROUTER_API_KEY.")
            else:
                website_info = get_website_info(question)

                with st.expander("View searched website text"):
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
- Answer in simple student-friendly language.
- Use bullet points.
- Include the source website URL.
- If the website text has partial information, still answer using what is available.
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