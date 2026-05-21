import os
import streamlit as st
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.title("PSU Harrisburg Website Assistant")


def scrape_psu_pages():
    start_url = "https://harrisburg.psu.edu/"
    visited = set()
    urls_to_visit = [start_url]
    all_text = ""

    max_pages = 20

    while urls_to_visit and len(visited) < max_pages:
        url = urls_to_visit.pop(0)

        if url in visited:
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            visited.add(url)

            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(" ", strip=True)
            all_text += f"\n\nSource: {url}\n{text[:4000]}"

            for link in soup.find_all("a", href=True):
                href = link["href"]

                if href.startswith("/"):
                    href = "https://harrisburg.psu.edu" + href

                if href.startswith("https://harrisburg.psu.edu") and href not in visited:
                    urls_to_visit.append(href)

        except Exception as e:
            all_text += f"\nCould not read {url}: {e}"

    return all_text


question = st.chat_input("Ask something about PSU Harrisburg")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PSU Harrisburg website..."):

            if not os.getenv("OPENROUTER_API_KEY"):
                st.error("Missing OPENROUTER_API_KEY in your .env file.")

            else:
                website_info = scrape_psu_pages()

                llm = ChatOpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url=os.getenv("OPENROUTER_BASE_URL"),
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful assistant for Penn State Harrisburg.

Only answer questions related to Penn State Harrisburg.

If the user asks something that is not related to Penn State Harrisburg, say:
"I can only answer questions related to Penn State Harrisburg."

Use only the website information below to answer.

Give the answer in bullet points.

At the end of the answer, include the related source website URL if it is available.

Website information:
{website_info}

User question:
{question}
"""

                try:
                    response = llm.invoke([
                        HumanMessage(content=prompt)
                    ])

                    st.write(response.content)

                except Exception as e:
                    st.error("OpenRouter error. Check your API key, credits, or model name.")
                    st.write(e)