
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI



BASE_URL = "https://api.avalai.ir/v1"
llm = ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL, temperature=0.2, max_tokens=2048)
