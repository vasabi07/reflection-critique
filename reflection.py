from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import MessagesState,StateGraph,START,END
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage,SystemMessage


class reflectionState(TypedDict):
    critique: str
    Merits: str
    Recommendations: str


# class State(MessagesState):
#     reflections: reflectionState

llm = ChatOpenAI(model="gpt-4o")

def Generate(state: MessagesState):
    """you work with a critique. generate responses for queries """
    # last_message = state["messages"][-1]
    response = llm.invoke(state["messages"])

    return {"messages": response}


def reflection(state: MessagesState):
    """you are a critique. analyse the message and critique it"""
    last_message = state["messages"][-1]
    reflection_prompt = f"""
    Analyze the following AI response and provide structured feedback:
    
    Response to analyze: {last_message.content}
    
    Provide:
    1. A critique of the response
    2. The merits of the response
    3. Specific recommendations for improvement
    """
    llm_with_structured = llm.with_structured_output(reflectionState)
    response = llm_with_structured.invoke(reflection_prompt)

    formatted_feedback = f"""
    Critique: {response['critique']}
    Merits: {response['Merits']}
    Recommendations: {response['Recommendations']}

    Based on this feedback, please generate an improved response.
    """
    return {"messages":[*state["messages"]]+[HumanMessage(content=formatted_feedback)]}

def should_continue(state: MessagesState):
    if len(state["messages"])> 6:
        return END
    else:
        return "reflection"

workflow = StateGraph(MessagesState)
workflow.add_node("generate",Generate)
workflow.add_node("reflection",reflection)
workflow.add_edge(START,"generate")
workflow.add_conditional_edges("generate",should_continue)
workflow.add_edge("reflection","generate")

graph = workflow.compile()

if __name__ == "__main__":
    system_message = """you are a agent who works with a critque. 
    generate responses as per user queries and work on giving better response with the critique. """
    response = graph.invoke({"messages":[SystemMessage(content=system_message)]+ [HumanMessage(content="write a 250 word essay on roger federer's achievements")]})
    print(response)

