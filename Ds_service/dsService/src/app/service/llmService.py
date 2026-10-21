import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .Expense import Expense


class LLMService:

    def __init__(self):
        load_dotenv()

        self.apiKey = os.getenv("GEMINI_API_KEY")

        if not self.apiKey:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert expense SMS extraction algorithm. "
                    "Extract only relevant information from the bank transaction SMS. "
                    "If you do not know the value of an attribute, return null."
                ),
                ("human", "{text}")
            ]
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=self.apiKey,
            temperature=0
)

        self.runnable = (
            self.prompt
            | self.llm.with_structured_output(schema=Expense)
        )

    def runLLM(self, message):
        return self.runnable.invoke({"text": message})