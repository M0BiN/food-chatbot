import sqlite3
import Levenshtein
from langchain_core.tools import tool
from utilities import document_search
from pydantic import BaseModel

# atexit.register(lambda: connection.close())

@tool
def retrieve_from_doc(query: str) -> list[str]:
    """
    Search for food-related information inside the documents.

    This tool is specifically designed for performing searches within a document database 
    to retrieve relevant information about food-related topics, such as ingredients, preparation methods, 
    or storage instructions. 

    **Usage Guidelines:**
    - The `query` string should be formulated in a way that increases the likelihood of finding matches in the database. 
      For example, use specific terms and phrases (e.g., "storage conditions for olive oil" or "nutritional value of avocados").

    **Behavior:**
    - The tool performs a hybrid search and returns up to three results with a relevance score above 0.6.
    - If no relevant matches are found, it will return ["NO RESULT!"].

    Args:
        query (str): A detailed, document-style query about food-related topics.

    Returns:
        list[str]: A list of matching texts from the database or ["NO RESULT!"] if no relevant matches are found.
    """
    result = document_search(query)
    if len(result) > 0:
        return [item['text'] for item in result]
    return ["NO RESULT!"]



@tool
def available_food_search(
    food_name: str = None, 
    restaurant_name: str = None) -> list[str]:
    """
    Searches the local database for available food items. Use this tool when the user is asking for a list of foods they can order locally.

    Args:
        food_name (str, optional): The name of the food to search for. Defaults to None.
        restaurant_name (str, optional): The name of the restaurant to search for. Defaults to None.

    Returns:
        list[str]: A list of available foods.
    """
    max_distance = 1
    connection = sqlite3.connect('food_orders.db')

    cursor = connection.cursor()
    cursor.execute("SELECT id, food_name, food_category, restaurant_name, price FROM foods")
    results = cursor.fetchall()

    matches = []
    for food_id, db_food_name, food_category, db_restaurant_name, db_price in results:
        food_name_distance = float('inf')
        restaurant_name_distance = float('inf')

        if food_name:
            food_name_distance_1 = Levenshtein.distance(food_name.lower(), db_food_name.lower(), weights=(0, 1, 1))
            food_name_distance_2 = Levenshtein.distance(food_name.lower(), db_food_name.lower(), weights=(1, 0, 1))
            food_name_distance_3 = Levenshtein.distance(food_name.lower(), db_food_name.lower(), weights=(1, 1, 1))
            food_name_distance = min(food_name_distance_1, food_name_distance_2, food_name_distance_3)
            

        if restaurant_name:
            restaurant_name_distance_1 = Levenshtein.distance(restaurant_name.lower(), db_restaurant_name.lower(), weights=(0, 1, 1))
            restaurant_name_distance_2 = Levenshtein.distance(restaurant_name.lower(), db_restaurant_name.lower(), weights=(1, 0, 1))
            restaurant_name_distance_3 = Levenshtein.distance(restaurant_name.lower(), db_restaurant_name.lower(), weights=(1, 1, 1))
            
            restaurant_name_distance = min(restaurant_name_distance_1, restaurant_name_distance_2, restaurant_name_distance_3)
            
        if food_name and restaurant_name:
            if food_name_distance <= max_distance and restaurant_name_distance <= max_distance:
                matches.append({
                    'id': food_id,
                    'food_name': db_food_name,
                    'food_category': food_category,
                    'restaurant_name': db_restaurant_name,
                    'price': db_price,
                    'edit_distance': min(food_name_distance, restaurant_name_distance)
                })
        elif food_name:
            if food_name_distance <= max_distance:
                matches.append({
                    'id': food_id,
                    'food_name': db_food_name,
                    'food_category': food_category,
                    'restaurant_name': db_restaurant_name,
                    'price': db_price,
                    'edit_distance': food_name_distance
                })
        elif restaurant_name:
            if restaurant_name_distance <= max_distance:
                matches.append({
                    'id': food_id,
                    'food_name': db_food_name,
                    'food_category': food_category,
                    'restaurant_name': db_restaurant_name,
                    'price': db_price,
                    'edit_distance': restaurant_name_distance
                })

    matches.sort(key=lambda x: x['edit_distance'])
    connection.close()
    return matches

@tool
def cancel_order(order_id:int, phone_number:str):
    """
    Cancel an order if its status is 'preparation'.
    :param connection: SQLite database connection
    :param order_id: ID of the order to cancel
    :return: Result message
    """
    connection = sqlite3.connect('food_orders.db')
    cursor = connection.cursor()
    
    cursor.execute("SELECT status FROM food_orders WHERE id = ? AND person_phone_number = ?", (order_id,phone_number))
    result = cursor.fetchone()
    
    if result is None:
        return f"Order ID {order_id} from {phone_number} does not exist."
    
    current_status = result[0]
    
    if current_status == "preparation":
        cursor.execute("UPDATE food_orders SET status = 'canceled' WHERE id = ?", (order_id,))
        connection.commit()
        connection.close()
        return f"Order ID {order_id} from {phone_number} has been successfully canceled."
    else:
        connection.close()
        return f"Order ID {order_id} from {phone_number} cannot be canceled as it is in '{current_status}' status."

@tool
def comment_order(order_id:int, person_name:str ,comment:str):
    """
    Add or overwrite a comment for an order.
    :param connection: SQLite database connection
    :param order_id: ID of the order to comment on
    :param comment: The comment to add or overwrite
    :return: Result message
    """
    connection = sqlite3.connect('food_orders.db')
    cursor = connection.cursor()
    
    cursor.execute("SELECT id FROM food_orders WHERE id = ?", (order_id,))
    result = cursor.fetchone()
    
    if result is None:
        return f"Order ID {order_id} does not exist."
    
    cursor.execute("UPDATE food_orders SET comment = ? WHERE id = ?", (comment, order_id))
    connection.commit()
    connection.close()
    return f"Comment for Order ID {order_id} from {person_name} has been updated."


@tool
def check_order_status(order_id: int):
    """
    Retrieves the status of a specific order using its unique integer ID.

    This function connects to the database, queries the 
    'food_orders' table for the given `order_id`, and returns the current status 
    of the order. If the specified `order_id` does not exist, an appropriate 
    error message is returned.

    **Usage Guidelines:**
    - This function must be used exclusively when the `order_id` (integer) is provided.
    - It is designed specifically for retrieving the status of an order and should 
    not be used for other types of queries or modifications.

    :param order_id: (int) The unique integer identifier of the order whose status is to be checked.
    :return: A string indicating the current status of the order or an error message 
            if the order ID does not exist.
    """


    connection = sqlite3.connect('food_orders.db')
    cursor = connection.cursor()
    
    cursor.execute("SELECT status FROM food_orders WHERE id = ?", (order_id,))
    result = cursor.fetchone()
    connection.close()
    if result is None:
        return f"Order ID {order_id} does not exist."
    
    return f"Order ID {order_id} from is currently in '{result[0]}' status."





class CompleteOrEscalate(BaseModel):
    """
    🚀 **CompleteOrEscalate Tool**

    **Purpose:**  
    This tool finalizes a task or escalates a request when it cannot be fulfilled. It signals that:  
    - A task is successfully completed.  
    - A request is off-topic or unsupported.  
    - No solution was found after reasonable attempts.  

    **Usage:**  
    - **Finalizing a Task:**  
    - `cancel`: `False`  
    - `reason`: Brief summary of the completed task.  
    - **Escalating/Terminating:**  
    - `cancel`: `True`  
    - `reason`: Explanation of why escalation is needed.  

    **Fields:**  
    - `cancel` (bool): `False` for completion, `True` for escalation.  
    - `reason` (str): Explanation for finalizing or escalating.  
    """
    cancel: bool
    reason: str
