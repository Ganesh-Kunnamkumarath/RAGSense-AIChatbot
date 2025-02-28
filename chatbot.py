import os
import gradio as gr
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

#  CONFIG
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "chatbot-index"

#  INIT PINECONE
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' does not exist. Please create it first.")
    exit()
index = pc.Index(INDEX_NAME)

#  INIT EMBEDDINGS & VECTORSTORE
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index, embedding_model, text_key="text")

#  INIT LLM & RAG CHAIN
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=vector_store.as_retriever(),
)

#  CHAT FUNCTIONS
def user(user_message, history):
    """
    Appends the user's message to the conversation history.
    Returns an empty string to clear the input box, plus the updated history.
    """
    if not user_message.strip():
        return "", history
    return "", history + [[user_message, None]]

def bot(history):
    """
    Retrieves the last user message from history, queries RAG, and appends the bot's response.
    If RAG says 'I don't know', fallback to the general LLM.
    """
    user_message = history[-1][0]  # last user query
    response = qa_chain.run(user_message)

    # Fallback to general knowledge if RAG is uncertain
    if "I don't know" in response or "I'm not sure" in response:
        fallback = "I couldn't find an answer in my knowledge base, but here's what I think...\n\n"
        response = fallback + llm.invoke(user_message).content

    # Update the last tuple in history with the bot's response
    history[-1][1] = response
    return history


#  BUILD GRADIO UI
with gr.Blocks() as demo:
    gr.Markdown("## AI Chatbot with RAG + Fallback")
    
    # Display conversation history
    chatbot = gr.Chatbot(label="Chat History")
    
    with gr.Row():
        msg = gr.Textbox(
            lines=2, 
            placeholder="Ask me anything...Press Shift + Enter to Send", 
            label="Your Message",
            scale=8
        )
        with gr.Column(scale=2):
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")

    # When user clicks "Send" → run user() → then bot()
    send.click(
        fn=user, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot]
    ).then(
        fn=bot, 
        inputs=chatbot, 
        outputs=chatbot
    )

    # When user presses Enter in the textbox → same flow
    msg.submit(
        fn=user, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot]
    ).then(
        fn=bot, 
        inputs=chatbot, 
        outputs=chatbot
    )

    # Clear conversation
    clear.click(lambda: [], None, chatbot)

demo.launch()


# Improvements Before Deployment

# To make the chatbot more robust, consider these enhancements:

# ✅ 1. Add Memory for Contextual Conversations
# Right now, the chatbot answers one query at a time. Adding memory will allow it to remember past interactions within a session.
# 📌 How? Use ConversationBufferMemory in langchain.chains.ConversationalRetrievalChain.

# ✅ 2. Improve Retrieval Performance
# Try Hybrid Search (Vector + Keyword Search)
# Pinecone supports hybrid search (mix dense embeddings + keyword-based search).
# Helps find better matches when exact wording differs.

# ✅ 3. Optimize User Input Handling
# Trim whitespace, prevent empty queries.
# Allow follow-up questions without repeating full context.
# Autoscroll chat to the latest message for better UX.

# ✅ 4. Logging & Analytics
# Track queries, fallback triggers, retrieval success rate.
# Store logs for debugging and model improvement.
