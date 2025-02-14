from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm
from tools import available_food_search, CompleteOrEscalate
from langchain_core.tools import tool

@tool
class ToSuggestionFood(BaseModel):
    """
    A tool to analyze user input and extract relevant criteria for food suggestions.

    **Purpose:**
    - Process user input to extract key preferences for food suggestions.

    **Usage:**
    - Fill the fields based on the user's input.

    **Field Descriptions:**
    - `criteria`: The user's specific craving or food preference (e.g., spicy, fast food, vegetarian).
    - `context`: Additional context such as price range, specific ingredients, or location.

    At least one field must be filled! Leave fields blank if not applicable.
    """
    criteria: str = Field(description="The user's specific craving or food preference (e.g., spicy, fast food, chinese).")
    context: Optional[str] = Field(description="Additional context like price range, location, or special ingredients.")

class Food(BaseModel):
    """
    Represents a food item recommended to the user.

    **Fields:**
    - `name`: The name of the dish.
    - `price`: The confirmed price of the dish.
    - `restaurant`: The name of the restaurant where the dish is available.
    """

    name: str = Field(description="The name of the recommended dish.")
    price: str = Field(description="The confirmed price of the dish.")
    restaurant: str = Field(description="The restaurant where the dish is available.")

class FoodRecommendation(BaseModel):
    """
    A structured tool for recommending food based on user preferences.
    
    **Purpose:**
    - Suggest available food options that match user criteria.
    - Provide a reflection on the strengths and weaknesses of the recommendations.
    - Generate search queries for refining or improving the recommendations.
    
    **Fields:**
    - `suggested_foods`: A list of available food options, including their name, price, and reason for recommendation.
    - `reflection`: Analysis of the recommendations, noting potential mismatches or areas for improvement.
    - `search_queries`: 1-3 queries to validate availability, refine suggestions, or find better alternatives.
    """
    suggested_foods: List[Food] = Field(
        description="A list of recommended available foods, each containing 'name', 'price', and 'restaurant'."
    )
    reflection: str = Field(
        description="Analysis of the food recommendations, highlighting weakness and possible mismatch to initial criteria."
    )
    search_queries: List[str] = Field(
        description="Only one **1** search queries to refine the recommendations, find better options"
    )

food_suggestion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "🍽️ **Food Recommendation Assistant**\n\n"
            "You are a highly specialized assistant responsible for **recommending only available food options** based on user preferences. "
            "Your response must **strictly follow** the structure of the `AnswerQuestion` tool by filling its fields accurately.\n\n"

            "🔍 **ORDER OF OPERATIONS (STRICTLY FOLLOW THIS):**\n"
            "1️⃣ **Extract key user preferences**, including:\n"
            "   - Taste (spicy, sweet, savory, etc.)\n"
            "   - Dietary restrictions (vegetarian, halal, high-protein, etc.)\n"
            "   - Cuisine type (Italian, Mexican, Japanese, etc.)\n"
            "   - Meal context (quick meal, fine dining, snack, etc.)\n"
            "   - Budget constraints, location, or specific ingredients.\n\n"

            "2️⃣ **Generate dish ideas internally based on provided context and criteria.**\n"
            "   - Prioritize common and relevant options that match user preferences.\n"
            "   - Suggest a few options to provide choices.\n\n"
            "   - dont add option which are not good to user request(even indirectly).\n"

            "3️⃣ **Verify availability using `available_food_search`.**\n"
            "   - **Only include dishes that are confirmed available.**\n"
            "   - If no matches are found, refine the search and retry.\n"
            "   - If multiple searches fail, proceed with the closest available alternatives.\n\n"
            "4️⃣ **Fill the `FoodRecommendation` tool as follows:**\n"
            "   - **suggested_foods:**\n"
            "     ✅ Provide a structured list of **available** food options.\n"
            "     ✅ Each entry must include:\n"
            "       - **name**: The dish name.\n"
            "       - **price**: The confirmed price from `available_food_search`.\n"
            "       - **restaurant**: The location where the dish is available.\n"
            "     ✅ Ensure all dishes meet the user's criteria (e.g., taste, dietary needs, cuisine, budget).\n\n"

            "   - **reflection:**\n"
            "     ✅ Critically analyze the recommendations.\n"
            "     ✅ Highlight potential limitations, such as:\n"
            "       - Mismatch with user preferences.\n"
            "       - Availability issues in specific locations.\n"
            "       - Price constraints or dietary concerns.\n"
            "     ✅ Mention any necessary refinements to improve accuracy.\n\n"

            "   - **search_queries (1-2 queries):**\n"
            "     ✅ Generate **targeted search queries** to refine or improve recommendations.\n"
            "     ✅ Focus on:\n"
            "       - Finding more **available** food options that match user preferences.\n"
            "       - Validating availability across more restaurants.\n"
            "       - Refining suggestions based on price, dietary restrictions, or taste profile.\n"
            "     ✅ Ensure search queries are **concise, specific, and effective.**\n\n"

            "⚡ **STRICT RULES TO FOLLOW:**\n"
            "✔️ **Always verify availability before making a recommendation.**\n"
            "✔️ **Never suggest a dish unless confirmed by `available_food_search`.**\n"
            "✔️ **Use `AnswerQuestion` as the sole response format—do not provide direct recommendations outside this structure.**\n"
            "✔️ **Maintain professional yet engaging language, avoiding technical process explanations.**\n"
            "✔️ **Ensure responses are structured, relevant, and user-friendly.**\n"
            "✔️ **Try to understand and guess what is the user looking for.**\n"
            "✔️ **Make sure the foods you offer are varied.**\n\n"
            "📌 **REMEMBER:** Your response **must** be formatted correctly to fill the `AnswerQuestion` tool, ensuring accurate, validated, and structured food recommendations."
        ),
        ("user", "{messages}")
    ]
)







food_suggestion_safe_tools = [available_food_search, FoodRecommendation]
food_suggestion_sensitive_tools = []
food_suggestion_tools = food_suggestion_safe_tools + food_suggestion_sensitive_tools

food_suggestion_runnable = food_suggestion_prompt | llm.bind_tools(
    food_suggestion_tools 
)


class ReviseFoodRecommendation(FoodRecommendation):
    """
    A tool for refining and improving food recommendations based on critique and new insights.

    **Purpose:**
    - Modify the initial food recommendations to better match user preferences.
    - Address weaknesses in the original list, such as availability, pricing, or dietary mismatches.
    - Provide a reflection explaining why changes were made.
    - Generate refined search queries to further optimize the recommendations.

    **Fields:**
    - `suggested_foods`: A revised list of food options, ensuring better alignment with the user’s criteria.
    - `reflection`: A critical analysis explaining why adjustments were made and how they improve the recommendations.
    - `search_queries`: Only 1 refined search queries for further validation or enhancement of the recommendations.
    """



# Food Critique and Revision Prompt
food_revision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "🔎 **Food Recommendation Review & Refinement**\n\n"
            "You are responsible for improving previous food recommendations by analyzing critique and search results.\n\n"

            "🎯 **TASK OBJECTIVE:**\n"
            "You will receive a list of food recommendations, along with their reflection and search queries. "
            "Your task is to **revise the recommendations**, removing weak choices and replacing them with better alternatives "
            "that better fit the user’s preferences.\n\n"

            "🔍 **HOW TO REVISE THE FOOD RECOMMENDATIONS:**\n"
            "1️⃣ **Analyze the Critique:**\n"
            "   - Identify weak points in the original food recommendations.\n"
            "   - Pay attention to mismatches in price, cuisine, dietary preferences, or availability.\n"
            "   - Consider whether better alternatives exist based on the search results.\n\n"

            "2️⃣ **Revise the Food List:**\n"
            "   - Remove any dishes that do not fully align with the user's request.\n"
            "   - Replace them with **better options** that are:\n"
            "     ✅ More relevant in terms of taste, cuisine, or dietary needs.\n"
            "     ✅ More budget-friendly (if price was a concern).\n"
            "     ✅ More widely available at verified restaurants.\n"
            "   - Keep the final list **concise** (3-5 items) and ensure **every item is available**.\n\n"

            "3️⃣ **Provide a Critical Reflection:**\n"
            "   - Explain **why** the previous recommendations were changed.\n"
            "   - Highlight **how the revised list better fits the user’s needs.**\n"
            "   - Mention any remaining limitations (e.g., limited availability in certain locations).\n\n"

            "4️⃣ **Generate Search Queries for Further Improvement:**\n"
            "   - Create **1-2 refined search queries** to find even better food options if needed.\n"
            "   - These should focus on validating availability, refining the cuisine choice, or finding better price points.\n\n"

            "⚡ **WHEN TO CALL `CompleteOrEscalate`:**\n"
            "You must determine whether the revision process is successful or needs escalation:\n\n"
            "✅ **Call `CompleteOrEscalate(cancel=False, reason=...)` when:**\n"
            "   - The revised recommendations are finalized and fully meet the user’s request.\n"
            "   - The user’s preferences are satisfied with available food options.\n\n"
            "❌ **Call `CompleteOrEscalate(cancel=True, reason=...)` when:**\n"
            "   - No valid food recommendations are available after multiple refinement attempts.\n"
            "   - The user’s request is out of scope or cannot be fulfilled.\n"
            "   - The search did not return useful results, and no reasonable adjustments can be made.\n\n"

            "⚡ **HOW TO STRUCTURE YOUR OUTPUT:**\n"
            "You must fill and call the `ReviseFoodRecommendation` tool as follows:\n\n"

            "   - **suggested_foods:**\n"
            "     ✅ A list of revised food options (each including **name, price, and restaurant**).\n\n"

            "   - **reflection:**\n"
            "     ✅ A short analysis explaining **why** the changes were made and **how** they improve the recommendations.\n\n"

            "   - **search_queries:**\n"
            "     ✅ 1-2 queries to **further refine** the recommendations.\n\n"

            "   - **Preserve User Input Fields (`criteria` & `context`)**\n"
            "       - You will receive `criteria` (user's original food preference) and `context` (additional details like price range, location, or ingredients).\n"
            "       - **DO NOT modify, reinterpret, or change these fields in any way.** They must remain exactly as provided by the user.\n"
            "       - Always pass them forward **unchanged** in every step of the process.\n"
            "       - If additional details are needed, use separate variables but never overwrite the original values.\n"
            "       - Any refinement to suggestions should be based on these values without altering them.\n\n"

            "📌 **IMPORTANT RULES:**\n"
            "✔ **Always prioritize user preferences when making changes.**\n"
            "✔ **Never include unavailable dishes in the revised list.**\n"
            "✔ **Ensure the final list is practical, well-structured, and concise and varied (if possible).**\n"
            "✔ **Dont worry about location of restaurant if not mentioned by user**\n"
            "✔ **Never remove options which are available and align with user criteria! **\n"
        ),
        ("user", "{messages}")
    ]
)


food_revision_safe_tools = [available_food_search, ReviseFoodRecommendation]
food_revision_sensitive_tools = []
food_revision_tools = food_revision_safe_tools + food_revision_sensitive_tools

food_revision_runnable = food_revision_prompt | llm.bind_tools(
    food_revision_tools + [CompleteOrEscalate] 
)
