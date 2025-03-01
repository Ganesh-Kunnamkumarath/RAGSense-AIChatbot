# 🚀 RAGSense – AI-Powered Knowledge Chatbot  

RAGSense-AIChatbot is an **AI-driven chatbot** that leverages **Retrieval-Augmented Generation (RAG)** to provide smart, context-aware responses. It integrates **Gradio** for an interactive UI, **Pinecone** for efficient vector storage, and **LangChain** to streamline LLM-based queries.

## ✨ Features  
✔️ **Smart Query Handling** – Retrieves and generates accurate responses.  
✔️ **I-Don’t-Know Optimization** – Provides helpful fallback suggestions instead of generic "I don’t know" responses.  
✔️ **Memory Integration** – Allows follow-up questions without repeating context.  
✔️ **Optimized UX** – Auto-scrolls chat, trims unnecessary whitespace, and prevents empty queries.  
✔️ **Easy Cloud Deployment** – Ready for hosting on **Hugging Face Spaces**.  


Demo link - https://huggingface.co/spaces/Ganesh-Kunnamkumarath/RAGSense-AIChatbot

---

## 🛠️ Tech Stack  

| Component      | Technology Used |
|---------------|----------------|
| **LLM**       | OpenAI GPT-4   |
| **Framework** | LangChain      |
| **Vector DB** | Pinecone       |
| **UI**        | Gradio         |
| **Hosting**   | Hugging Face Spaces |

---

## 💡 Thought Process Behind Development  

1. **Efficient Query Handling**  
   - We needed a chatbot that could **retrieve relevant context** rather than just generate text blindly.  
   - Implemented **RAG (Retrieval-Augmented Generation)** to fetch accurate responses.  

2. **Improving the "I Don’t Know" Scenario**  
   - Instead of replying "I don’t know," the bot now **provides feedback** and suggests alternative information.  

3. **Enhancing User Experience (UX)**  
   - **Whitespace trimming** and **empty query prevention** to ensure a smoother conversation.  
   - Implemented **auto-scroll** to keep the chat focused on the latest response.  
   - Allows **follow-up questions** while maintaining context, improving engagement.  

4. **Deployment Strategy**  
   - Used **Gradio’s `share=True`** for easy local testing.  
   - Deployed to **Hugging Face Spaces** for public access.  

---

## 🚀 How to Run Locally  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Ganesh-Kunnamkumarath/RAGSense-AIChatbot.git
cd RAGSense-AIChatbot

