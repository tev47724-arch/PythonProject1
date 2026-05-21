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


def scrape_psu_pages():
    urls = [
        "https://harrisburg.psu.edu/",
        "https://harrisburg.psu.edu/academics",
        "https://harrisburg.psu.edu/academic-programs",
        "https://harrisburg.psu.edu/college-business-administration",
        "https://harrisburg.psu.edu/school-science-engineering-technology",
        "https://harrisburg.psu.edu/school-public-affairs",
        "https://harrisburg.psu.edu/school-behavioral-sciences-and-education",
        "https://harrisburg.psu.edu/school-humanities",
        "https://harrisburg.psu.edu/admissions",
        "https://harrisburg.psu.edu/financial-aid",
        "https://harrisburg.psu.edu/campus-life",
        "https://harrisburg.psu.edu/student-life",
        "https://harrisburg.psu.edu/housing-and-food-services",
        "https://harrisburg.psu.edu/career-services",
        "https://harrisburg.psu.edu/advising",
        "https://harrisburg.psu.edu/contact-us",
        "https://www.registrar.psu.edu/academic-calendars/",
        "https://bulletins.psu.edu/programs/",
        "https://admissions.psu.edu/",
        "https://studentaid.psu.edu/",
        "https://liveon.psu.edu/",
        "https://libraries.psu.edu/",
        "https://canvas.psu.edu/",
        "https://lionpath.psu.edu/",
        "https://webaccess.psu.edu/",
        "https://it.psu.edu/",
        "https://foodservices.psu.edu/",
        "https://transportation.psu.edu/",
        "https://global.psu.edu/"
    ]

    all_text = ""

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            all_text += f"\n\nSource: {url}\n{text[:7000]}"

        except Exception as e:
            all_text += f"\n\nSource: {url}\nCould not read this page: {e}"

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
                website_info = scrape_psu_pages()

                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful assistant for Penn State Harrisburg.

You should answer questions about:
- Penn State Harrisburg
- Penn State academics
- admissions
- financial aid
- housing
- campus life
- student services
- academic calendars
- Canvas
- LionPATH
- PSU technology resources

If the user asks a question that is clearly unrelated to Penn State or PSU Harrisburg, say:
"I can only answer questions related to Penn State Harrisburg."

Use the website content below to answer.

Do not say you do not have access unless the answer is truly not found.

Give answers in bullet points.

Always include the most relevant source URL at the end.

Website content:
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