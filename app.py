
from langchain_core.messages import HumanMessage
from graphs.supergraph import supergraph
import chainlit as cl
import uuid
def get_label(node_name:str):
    return {"primary_assistant":"Primary Assistant",
    "generate":"Generate Response",
    "food_suggestion":"Food Suggester",
    "food_search":"Food Search",
    "doc_retrieval":"Doc Retrieval",
    "enter_content_grader":"Content Grader",
    "enter_generate":"Generate Response",
    "enter_web_search":"Web Search",
    "web_search":"Web Search",
    }.get(node_name, None)
    return ["primary_assistant", "generate", "food_suggestion", "food_search"]
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))

import asyncio

@cl.on_message
async def on_message(msg: cl.Message):
    try:
        cb = cl.LangchainCallbackHandler()
        final_answer = cl.Message(content="")
        config = {
            "configurable": {
                "user_info": "Ali Akbar",
                "thread_id": cl.user_session.get("thread_id"),
            }
        }

        async with cl.Step(name="Gemini", type="llm") as step:
            step.input = msg.content
            step.output = ""

            tasks = []  # Track all running async tasks

            async def async_stream(resume:bool):
                for streamed_msg, metadata in supergraph.stream(
                    None if resume else {"messages": [HumanMessage(content=msg.content)]},
                    stream_mode="messages",
                    config=config
                ):
                    yield streamed_msg, metadata  # Convert to async generator


            async def process_stream(resume:bool=False):
                async for streamed_msg, metadata in async_stream(resume):

                    if (
                        streamed_msg.content
                        and not isinstance(streamed_msg, HumanMessage)
                        and metadata["langgraph_node"] in ["primary_assistant", "generate", "food_suggestion", "food_search"]
                    ):
                        await final_answer.stream_token(streamed_msg.content)
                    else:
                        if(get_label(metadata["langgraph_node"])):
                            step.name = get_label(metadata["langgraph_node"])
                            await step.update()

            # Start the async task and track it
            stream_task = asyncio.create_task(process_stream())
            tasks.append(stream_task)
            
            try:
                await asyncio.gather(*tasks)  # Wait for all tasks to complete
            except asyncio.CancelledError:
                print("Cancelling all tasks...")
                for task in tasks:
                    task.cancel()  # Cancel each task
                await asyncio.gather(*tasks, return_exceptions=True)  # Ensure all are properly cancelled
        state = supergraph.get_state(config=config)

        
        if state.next:
            res = await cl.AskActionMessage(
            content="Are you sure?!",
            actions=[
                cl.Action(name="continue", payload={"value": "continue"}, label="✅ YES"),
                cl.Action(name="cancel", payload={"value": "cancel"}, label="❌ Cancel"),
            ],
        ).send()
            if res and res.get("payload").get("value") == "continue":
                pass
                # await cl.Message(
                #     content="Continue!",
                # ).send()
            else:
                supergraph.update_state(
                    config,
                    {"messages": [HumanMessage(
                            content=f"i changed my mind.",
                        )], "next":"leave_skill"},
                )
            stream_task = asyncio.create_task(process_stream(resume=True))
            tasks.append(stream_task)
            await asyncio.gather(*tasks)
    except Exception as e:
        print(e)
        await final_answer.stream_token("I'm sorry, but I couldn't process your request at this time. Could you try rephrasing or providing it in a different format?")

    await final_answer.send()  # Send the final response after streaming completes
