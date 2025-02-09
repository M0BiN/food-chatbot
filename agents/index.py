from pydantic import BaseModel, Field



class ToWebSearch(BaseModel):
    """
    A tool for performing a web search when the cleaned content is not relevant 
    or sufficient to answer the user's query.

    **Purpose:**
    - This tool is used when the `cleaned_content` does not provide adequate or relevant information to sufficiently address the `user_query`.
    - It retrieves additional information from the web to ensure the user's query can be answered effectively.

    **Usage Guidelines:**
    - Invoke this tool only if the `content_grader` determines that the `cleaned_content` is irrelevant or insufficient to answer the user's question.
    - Populate the `user_query` field with the original user query exactly as received, without any modifications.
    - Ensure the `user_query` reflects the user's intent to optimize the web search results.

    **Behavior:**
    - Use this tool whenever the `cleaned_content` is inadequate or irrelevant.
    - Do not attempt to modify or interpret the content; simply delegate to this tool to retrieve additional information.
    - Ensure the user's intent is preserved by passing the original query unchanged.
    """

    user_query: str = Field(
        description="The original user query, passed without modification, to guide the web search when the cleaned content is deemed irrelevant or insufficient."
    )