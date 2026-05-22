import os
import requests
import streamlit as st

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
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


def is_valid_psu_link(url):
    parsed = urlparse(url)

    return (
        parsed.scheme in ["http", "https"]
        and "harrisburg.psu.edu" in parsed.netloc
        and "#" not in url
        and not url.endswith((".pdf", ".jpg", ".png", ".zip"))
    )


def get_page_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    return soup.get_text(" ", strip=True), soup


def crawl_psu_website(start_url, max_pages=12):
    visited = set()
    pages_to_visit = [start_url]
    all_text = ""

    while pages_to_visit and len(visited) < max_pages:
        url = pages_to_visit.pop(0)

        if url in visited:
            continue

        try:
            text, soup = get_page_text(url)
            visited.add(url)

            all_text += f"""

SOURCE WEBSITE:
{url}

SCRAPED WEBSITE TEXT:
{text[:8000]}

"""

            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"])

                if is_valid_psu_link(full_url) and full_url not in visited:
                    pages_to_visit.append(full_url)

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
        with st.spinner("Searching PSU Harrisburg website..."):

            api_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")

            if not api_key:
                st.error("Missing OPENROUTER_API_KEY.")
            else:
                website_info = crawl_psu_website(
                    "https://harrisburg.psu.edu/",
                    max_pages=12
                )

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

Use ONLY the scraped PSU Harrisburg website text below.

Rules:
- Answer in simple bullet points.
- Include the source website URL.
- If only partial information is found, still answer using what is available.
- Do not make up information.

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