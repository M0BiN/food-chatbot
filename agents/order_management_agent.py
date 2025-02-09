from typing import Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import llm
from tools import cancel_order, comment_order, check_order_status, CompleteOrEscalate
from langchain_core.tools import tool
from tools import CompleteOrEscalate


@tool
class ToOrderManagement(BaseModel):
    """
    A tool for handling user operations related to managing orders, including canceling orders, 
    providing feedback, and checking order status.

    **Purpose:**
    - Automate order-related operations based on user input with strict adherence to required parameters.

    **Usage:**
    - Provide exactly the required fields based on the selected operation type.
    - Ensure inputs are valid and align with the described requirements for the operation.

    **Field Requirements by Operation:**
    - `cancel_order`:
        - `order_id` (int): The unique identifier for the user's order.
        - `phone_number` (str): The user's phone number for verification purposes.
    - `comment_order`:
        - `order_id` (int): The unique identifier for the user's order.
        - `person_name` (str): The name of the person providing the comment.
        - `comment` (str): The content of the feedback.
    - `check_order_status`:
        - `order_id` (int): The unique identifier for the user's order.

    **Field Descriptions:**
    - `operation`: Specifies the type of operation the user wants to perform.
    - `order_id`: (Required for all operations) The unique identifier for the user's order.
    - `phone_number`: (Required for `cancel_order`) The user's phone number for verification.
    - `person_name`: (Required for `comment_order`) The name of the person providing feedback.
    - `comment`: (Required for `comment_order`) The content of the user's feedback.

    """

    operation: Literal["cancel_order", "comment_order", "check_order_status"] = Field(
        description="The type of operation the user wants to perform (e.g., 'cancel_order', 'comment_order', 'check_order_status')."
    )
    order_id: int = Field(
        description="The unique identifier for the user's order. Required for all operations."
    )
    phone_number: Optional[str] = Field(
        default=None, description="The user's phone number for verification (required for 'cancel_order')."
    )
    person_name: Optional[str] = Field(
        default=None, description="The name of the person providing feedback (required for 'comment_order')."
    )
    comment: Optional[str] = Field(
        default=None, description="The content of the feedback (required for 'comment_order')."
    )

    class Config:
        json_schema_extra = {
            "examples": {
                "cancel_order": {
                    "operation": "cancel_order",
                    "order_id": 12345,
                    "phone_number": "09123456789"
                },
                "comment_order": {
                    "operation": "comment_order",
                    "order_id": 98765,
                    "person_name": "John Doe",
                    "comment": "The food was fantastic!"
                },
                "check_order_status": {
                    "operation": "check_order_status",
                    "order_id": 54321
                }
            }
        }



# Assistant for Order Management

order_management_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant specialized in managing user operations related to orders. "
            "Your sole responsibility is to call tools, retrieve their outputs, and respond to the user based on those outputs. "
            "You must not provide answers from your own knowledge or assumptions. "
            "Your tasks include canceling orders, checking the status of orders, and adding comments to orders. "
            "You are required to use tools to gather all information before responding. "
            "Never speculate or generate answers without first calling the appropriate tool."
            "\n\nAll user interactions must be finalized using the `CompleteOrEscalate` tool. "
            "This means after retrieving information, summarizing it humanly, and ensuring the user has their answer, "
            "you must call `CompleteOrEscalate` to confirm the completion of the task or escalate if needed."
            "\n\nWhen filling `CompleteOrEscalate.reason`, rewrite tool outputs in a natural, conversational tone. "
            "For example, if an order is being prepared, say so warmly instead of repeating raw tool output. "
            "If an order does not exist, acknowledge it politely and ask for clarification if needed."
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
            '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. '
            "Do not make up invalid tools or functions."
            "if some info for example phone number is not valid, dont hesitate and ask user to fix it!"
            "watch out for information validation!",
        ),
        ("placeholder", "{messages}"),
    ]
)









order_management_safe_tools = [check_order_status, comment_order]
order_management_sensitive_tools = [cancel_order]
order_management_tools = order_management_safe_tools + order_management_sensitive_tools
order_management_runnable = order_management_prompt | llm.bind_tools(
    order_management_tools + [CompleteOrEscalate], tool_choice="any"
)