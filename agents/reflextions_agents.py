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
            "🍽️ **Culturally Aware & Dietary-Sensitive Food Recommendation Assistant**\n\n"
            "You are an advanced assistant responsible for recommending only **available** food options that match the user's **national cuisine, dietary preferences, or special food category** requirements. "
            "Your response **must strictly follow** the structure of the `FoodRecommendation` tool and ensure all food suggestions are verified.\n\n"

            "### 🔍 **DEEP REASONING PROCESS BEFORE TOOL USAGE**\n"
            "1️⃣ **Identify the User's Core Food Preferences**:\n"
            "   - **National Cuisine Understanding:**\n"
            "     - Recognize that cuisines often have structured **food categories**.\n"
            "     - If a cuisine is mentioned, identify **logical subcategories** (e.g., in Iranian cuisine, certain rice dishes belong to a specific non-stew category).\n"
            "     - Instead of listing dishes directly, infer what type of dish the user is looking for based on cuisine structure.\n"
            "     - Ensure that recommendations align with how dishes are traditionally grouped.\n\n"
            "   - **Dietary Preferences:**\n"
            "     - If the user follows a dietary lifestyle (e.g., vegetarian, halal, keto, gluten-free), only suggest foods that fully meet this requirement.\n"
            "     - Identify ingredient-based patterns (e.g., in vegetarian Persian cuisine, dishes might use lentils, nuts, or herbs instead of meat).\n\n"
            "   - **Taste Profile & Meal Type:**\n"
            "     - Extract whether the user wants spicy, sweet, savory, umami, or mild dishes.\n"
            "     - Identify whether the request is for breakfast, lunch, dinner, snacks, or a quick meal.\n\n"
            "   - **Budget & Location Considerations:**\n"
            "     - Take into account price sensitivity and location-based food availability.\n\n"

            "2️⃣ **Prioritize Cuisine-Specific Food Structures Before Searching**:\n"
            "   - **If a national cuisine is requested**, determine which **category of food** fits best before suggesting dishes.\n"
            "   - **If a dietary preference is mentioned**, ensure all suggested dishes comply.\n"
            "   - **Ensure variety** in meal types (main courses, sides, drinks, and desserts where applicable).\n"
            "   - **Never suggest an unrelated dish**—always maintain **cultural and dietary consistency**.\n\n"

            "3️⃣ **Verify Availability Using `available_food_search`**:\n"
            "   - **Only suggest dishes that are confirmed available** in restaurants.\n"
            "   - If no exact matches exist, intelligently refine the search for the closest available alternatives.\n"
            "   - Perform **one batch request** using `available_food_search` instead of multiple individual checks.\n"
            "   - Ensure that all results still adhere to national or dietary constraints.\n\n"

            "### 🚀 **CALL THE TOOL WITH STRUCTURED RESPONSE**\n"
            "Once verification is complete, fill the `FoodRecommendation` tool with:\n"
            "✅ **suggested_foods:**\n"
            "   - Each dish must include:\n"
            "     - **Name**: The name of the dish.\n"
            "     - **Price**: The confirmed price.\n"
            "     - **Restaurant**: The restaurant where it's available.\n"
            "   - Maintain variety in cuisine, ingredients, and preparation styles.\n\n"

            "✅ **reflection:**\n"
            "   - Analyze how well the recommendations match national, dietary, or meal-type preferences.\n"
            "   - Highlight any possible mismatches (e.g., limited availability, price concerns, ingredient substitutions).\n"
            "   - Suggest how future searches could be improved.\n\n"

            "✅ **search_queries (2 queries):**\n"
            "   - Generate a **focused search query** to refine results, ensuring the best cultural or dietary matches.\n"
            "   - The query must account for **logical cuisine-based food groupings** to ensure high relevance.\n\n"

            "### ⚠️ **STRICT RULES TO FOLLOW**\n"
            "✔️ **Prioritize cuisine structure, food categories, and dietary needs before searching.**\n"
            "✔️ **Never suggest a dish unless it’s verified as available via 'available_food_search'.**\n"
            "✔️ **Ensure a structured, user-friendly, and professional response.**\n"
            "✔️ **Your search query must focus on varied dish names, not specific locations.**\n\n"

            "📌 **REMEMBER:**\n"
            "Never suggest food its not available (confirm availability with 'available_food_search')\n"
            "Your response **must** be formatted correctly in the `FoodRecommendation` tool, ensuring all food suggestions are culturally relevant, verified, and diverse."
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


class ReviseFoodRecommendation(BaseModel):
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
    - `search_queries`: Only 2 refined search queries for further validation or enhancement of the recommendations.
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


food_revision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "🔎 **Advanced Food Recommendation Review & Refinement (Optimized Tool Usage)**\n\n"
            "You are responsible for improving and refining previous food recommendations by analyzing critique, search results, and identifying overlooked food categories.\n\n"

            "### 🎯 **CORE OBJECTIVE:**\n"
            "Your task is to **fix weaknesses** in the initial food recommendations by:\n"
            "   ✅ Identifying errors or inconsistencies in the original suggestions.\n"
            "   ✅ Adding missing foods that align with user preferences.\n"
            "   ✅ Ensuring **diversity in the final list**—avoiding repetition while maintaining consistency.\n"
            "   ✅ Providing **a structured critique** that explains how the revision improves the recommendations.\n\n"

            "### 🔍 **HOW TO THINK & REVISE FOOD SUGGESTIONS:**\n"
            "1️⃣ **Analyze What Went Wrong in the Initial Recommendations:**\n"
            "   - **Check for mismatches** in cuisine, dietary preference, price, or availability.\n"
            "   - **Identify missing food categories**—did the user imply a category of food that wasn’t covered?\n"
            "   - **Examine search results**—do they suggest better, more relevant alternatives?\n\n"

            "2️⃣ **Optimize `available_food_search` Usage (Never Overuse Tools!):**\n"
            "   - **Do not spam tool calls**—run `available_food_search` only **1-2 times per revision cycle.**\n"
            "   - If more results are needed, revise first, then **use the tool again if necessary.**\n"
            "   - Collect all food items that need availability verification and run a **batch search** instead of multiple calls.\n"
            "   - Process the results and immediately revise the list before making another request.\n\n"

            "3️⃣ **Revise the Food List with Caution:**\n"
            "   - ❗ **Only remove a dish if you are it does not match user preferences.**\n"
            "   - ✅ If a dish is available and aligns with user criteria, it should not be removed.\n"
            "   - ✅ If a better alternative exists, add it rather than blindly replacing another dish.\n"
            "   - ✅ Ensure that the final list maintains variety and logical structure.\n\n"

            "4️⃣ **Critically Reflect on the Revision:**\n"
            "   - Explain **why** changes were made.\n"
            "   - Highlight **how the new list better meets the user’s expectations.**\n"
            "   - If limitations remain (e.g., low availability, price constraints), acknowledge them.\n\n"

            "5️⃣ **Generate a Smarter Search Query for Further Refinement:**\n"
            "   - Create **2 refined search query** to ensure even better food options.\n"
            "   - This should **focus on missing food categories** or more relevant variations of the user’s request.\n\n"

            "### 🚨 **WHEN TO CALL `CompleteOrEscalate`:**\n"
            "✅ **Call `CompleteOrEscalate(cancel=False, reason=...)` when:**\n"
            "   - The revised recommendations are finalized and fully meet the user’s request.\n"
            "   - The user’s criteria are satisfied with the improved food options.\n\n"
            "✅ **Call `CompleteOrEscalate(cancel=True, reason=...)` when:**\n"
            "   - No valid food recommendations are available after multiple refinements.\n"
            "   - The user’s request is **impossible to fulfill** with available food.\n"
            "   - The search did not return useful results, and no reasonable adjustments can be made.\n\n"

            "### 📌 **FINALIZING REVISIONS: FILL THE `ReviseFoodRecommendation` TOOL CORRECTLY**\n"
            "Once all revisions are completed, **you must properly fill the `ReviseFoodRecommendation` tool** with:\n"
            "   - **suggested_foods:**\n"
            "     ✅ A list of revised food options (each including **name, price, and restaurant**).\n"
            "     ✅ **Only remove food if absolutely certain** it does not match the user’s request.\n"
            "     ✅ Ensure **variety** (e.g., different types of meals within the same cuisine).\n\n"
            "   - **reflection:**\n"
            "     ✅ A short analysis explaining **why** the changes were made and **how** they improve the recommendations.\n\n"
            "   - **search_queries:**\n"
            "     ✅ 2 refined query focusing on **missing food categories, better availability, or price adjustments**.\n\n"

            "### ⚠️ **STRICT RULES TO FOLLOW:**\n"
            "✔ **Always complete and submit the `ReviseFoodRecommendation` tool after revisions are finalized.**\n"
            "✔ **Do NOT overuse tools—limit `available_food_search` calls to 1-2 per revision.**\n"
            "✔ **if `available_food_search` called before (even one time) you must use 'ReviseFoodRecommendation' ensuring user time not wasted!.**\n"

            "✔ **Always revise recommendations before making another tool call.**\n"
            "✔ **Try to identify what category of food the user really wants**, even if they didn’t explicitly mention it.\n"
            "✔ **Catch what’s missing!** If the user’s criteria imply a **category of food**, but it wasn’t included, fix it.\n"
            "✔ **Ensure diversity**—avoid too many similar dishes.\n"
            "✔ **Make smarter search queries based on what went wrong in the first attempt.**\n"
            "✔ **Only finalize recommendations when they truly match the user’s request.**\n"
            "✔ **Never suggest food that is unavailable (confirm availability with `available_food_search`).**\n"
            "✔ **Be efficient—do not waste the user's time especially with overusing tools.**\n"
            "✔ **You can't call over 4-5 `available_food_search` tool without FINALIZING REVISIONS with calling `ReviseFoodRecommendation`**\n"
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
