import os
import streamlit as st
import requests

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.title("PSU Harrisburg Website Assistant")

PSU_PROGRAMS_INFO = """
Source: https://harrisburg.psu.edu/academic-programs

Penn State Harrisburg offers undergraduate and graduate programs through several schools:

- School of Business Administration
- School of Science, Engineering, and Technology
- School of Public Affairs
- School of Behavioral Sciences and Education
- School of Humanities

Examples of program areas include:
- Business
- Engineering
- Information Sciences and Technology
- Computer Science
- Cybersecurity
- Criminal Justice
- Education
- Psychology
- Communications
- Public Policy
- Humanities
"""


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

            all_text += f"\n\nSource: {url}\n{text[:5000]}"

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
                st.error("Missing OPENROUTER_API_KEY")

            else:

                website_info = PSU_PROGRAMS_INFO + scrape_psu_pages()

                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful assistant for Penn State Harrisburg.

You ONLY answer questions related to:
- Penn State Harrisburg
- PSU academics
- admissions
- housing
- campus life
- student services
- financial aid
- Canvas
- LionPATH
- PSU technology resources

If the user asks unrelated questions, say:
"I can only answer questions related to Penn State Harrisburg."

You MUST answer using the website information below.

Do NOT say:
"I do not have access"

If information exists in the website content, answer it.

Give detailed bullet point answers.

Always include a source URL at the end.

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

                    st.error("OpenRouter error")
                    st.write(e)