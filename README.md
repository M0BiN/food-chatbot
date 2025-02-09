# 🍽️ Food Chatbot - AI Food Assistant 🤖
A smart AI-powered chatbot for **food recommendations, order management, and food information retrieval** using **LangGraph, LangChain, and Google Gemini AI**.

---

## 🚀 Features
✅ **Order Management** – Track and manage food orders.  
✅ **Food Search** – Find available dishes in different restaurants.  
✅ **Food Recommendations** – Get smart food suggestions based on preferences.  
✅ **Document Retrieval** – Answer food-related questions using a knowledge base.  
✅ **Multi-agent Architecture** – Uses specialized AI agents for efficient processing.  

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/M0BiN/food-chatbot.git
cd food-chatbot
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set up environment variables
Create a .env file and add:
```bash
GOOGLE_GEMINI_API_KEY=your_api_key_here
DATABASE_URL=mysql://user:password@localhost/foodchat
```
### 4️⃣ Run the chatbot
```bash
python main.py
```

### 📂 Project Structure
📂 food-chatbot
├── 📂 agents                  # AI Agents handling specific tasks
│   ├── doc_retrieval_agent.py
│   ├── order_management_agent.py
│   ├── food_search_agent.py
│   ├── food_suggestion_agent.py
│   ├── summarize_conversation_agent.py
├── 📂 graphs                  # LangGraph sub-graphs
│   ├── part_1_graph.py
│   ├── part_2_graph.py
│   ├── part_3_graph.py
│   ├── part_4_graph.py
├── 📂 tools                   # Helper functions & APIs
│   ├── CompleteOrEscalate.py
│   ├── utility_functions.py
│   ├── config.py
│   ├── database.py
│   ├── logger.py
│   ├── api_clients.py
├── 📂 tests                   # Unit tests
│   ├── test_order_management.py
│   ├── test_food_search.py
│   ├── test_conversation.py
│   ├── test_suggestions.py
├── main.py                    # Entry point
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
└── .env                        # Environment variables


### 📢 Contribution
Contributions are welcome! Please open an issue or submit a pull request. 😊