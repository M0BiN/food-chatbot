
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_google_genai import ChatGoogleGenerativeAI


llm_maxtokens = 512
llm_temperature = 0.01







gemini_llm_params = {
        'temperature': llm_temperature,
        'max_output_tokens': llm_maxtokens
}

gemini_safe = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
}

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", config=gemini_llm_params, safety_settings=gemini_safe)
