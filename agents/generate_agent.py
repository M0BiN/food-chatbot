
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm

class ToGenerate(BaseModel):
    """
    A tool for generating an answer to the user's query based on cleaned and structured content.

    **Purpose:**
    - This tool is invoked to generate a concise and accurate answer to the `user_query` using the provided `cleaned_content`.
    - It must only be used when the `content_grader` determines that the `cleaned_content` is relevant and sufficient.

    **Usage Guidelines:**
    - Populate the `cleaned_content` field with the refined, structured text that is sufficient to answer the `user_query`.
    - Populate the `user_query` field with the original query to ensure the generated answer is contextually accurate.
    - Ensure the tool is invoked only when the content is deemed relevant and adequate for answering the query.
    - Avoid invoking this tool if the `cleaned_content` is irrelevant or incomplete.
    """

    cleaned_content: str = Field(
        description="The cleaned and structured content that serves as the basis for answering the user's query."
    )
    user_query: str = Field(
        description="The original user query that must be answered using the cleaned content."
    )


# Generate assistant

generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for generating the best possible answer to the user's query using cleaned and structured content. "
            "Your task is to focus entirely on generating a concise and accurate answer to the `user_query` using the provided `cleaned_content`. "
            "Do not deviate from the provided content or make assumptions beyond the given information. "
            "You must use the `cleaned_content` as the sole source of truth for answering the query. "
            "\n\n### Guidelines:"
            "\n1. **Use the Cleaned Content:**"
            "\n   - Ensure your answer directly addresses the `user_query` using only the `cleaned_content`."
            "\n   - Do not mention or imply that your answer is based on provided content."
            "\n\n2. **Focus on Answer Quality:**"
            "\n   - Generate responses that are concise, accurate, and fully aligned with the user's intent."
            "\n   - Avoid adding unnecessary details or information outside the scope of the cleaned content."
            "\n\n3. **Tone and Style:**"
            "\n   - Maintain a polite, professional, and approachable tone in your responses."
            "\n   - Use clear and concise language to ensure your answer is easy to understand."
            "\n   - For food-related topics, you may incorporate a friendly touch, such as sparing use of food-related emojis like 🍕🥗."
        ),
        ("placeholder", "{messages}"),
    ]
)




generate_safe_tools = []
generate_sensitive_tools = []
generate_tools = generate_safe_tools + generate_sensitive_tools
generate_runnable = generate_prompt | llm