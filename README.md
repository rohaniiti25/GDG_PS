#  ---------------------- PS1 ---------------------- 
#  ---------------------- Task 1 ---------------------- 

# Pure LSTM Stock Forecast (7-Day)

## What this is

This repo is a **straightforward stock price forecasting experiment using a pure LSTM model** — no ARIMA, no hybrid tricks. The goal was to see how far we can get using **just deep learning**, clean preprocessing, and sensible training choices.

It’s built to be easy to follow and easy to experiment with.

---

## What the model does

Given historical stock data, the model:

* learns patterns from **log returns** (not raw prices)
* looks at the **last 60 trading days**
* predicts the **next 7 days of returns at once**
* converts those returns back into **future prices**

The final output is a clean **7-day price forecast** with a simple visualization.

---

## High-level workflow

1. Load stock data from CSV
2. Sort by date and select a single company
3. Convert prices → log returns
4. Standardize returns
5. Create rolling sequences (60 → next 7 days)
6. Train a pure LSTM model
7. Predict future returns
8. Convert returns back to prices
9. Plot and print results

---

## Model details (kept simple)

* **Architecture**: 2-layer LSTM
* **Hidden size**: 48
* **Input**: 60-day return sequence
* **Output**: 7-day return vector (direct multi-step prediction)
* **Loss**: Smooth L1 Loss (works better for noisy financial data)
* **Optimizer**: Adam
* **Early stopping**: Yes (to avoid overfitting)

No ARIMA, no ensembles — just LSTM.

---

## Why log returns?

* More stable than raw prices
* Easier for the model to learn
* Common practice in quantitative finance

We standardize returns instead of using MinMax scaling to handle outliers better.

---

## Dataset

* CSV file with columns like: Date, Company, Close
* Works with any stock as long as the format is consistent
* Automatically picks **AAPL** if present, otherwise uses the first company found

---

## What works well

* Training is stable
* Early stopping prevents overfitting
* Direct multi-step prediction avoids recursive error buildup
* Output is easy to interpret

---

## Known limitations

* No external factors (news, volume, fundamentals)
* Assumes past patterns repeat
* Not meant for real trading decisions

This is a learning and experimentation project, not financial advice.

---

## What this project shows

* Understanding of time-series preprocessing
* Practical use of LSTM for sequence-to-sequence prediction
* Awareness of financial data quirks (returns, noise, scaling)
* Ability to build an end-to-end ML pipeline


#  ---------------------- Task 2 ---------------------- 

🇮🇳 Indian Stock Market AI Assistant

This is an AI-powered stock market assistant for Indian stocks 🇮🇳📊
You ask questions like “Why is Reliance falling today?” and it gives you a clean, structured analysis using:

live market data

latest news

AI reasoning (LLM + vector search)

Built mainly as a learning + experimentation project.

🤔 What does this thing do?

You type a normal human question like:

Why is TCS stock going down today?

Trend of Infosys share price

When will Reliance recover?

And it:

Understands your intent

Detects the Indian company

Fetches live market data

Pulls latest news

Explains everything in simple markdown format

🧠 Core Features

📊 Live Indian stock prices (BSE)

📰 Latest market news (Economic Times + Moneycontrol)

🤖 LLM-powered explanations

🔍 Vector search on news articles

🧭 Intent detection (why / when / trend)

🖥️ Simple Gradio web UI

⏳ Loading spinner + progress status (feels cool 😄)

⚙️ Tech Stack

Python 🐍

yFinance (market data)

LangChain

OpenAI-compatible LLM

HuggingFace Embeddings

FAISS (vector search)

Gradio (UI)

📂 How It Works (Simple Flow)

User asks a question

AI extracts:

intent

company name

BSE ticker

Fetches live or recent market data

Pulls latest financial news

Builds a vector database

Retrieves relevant context

AI generates a structured markdown explanation

🧪 Output Format

The AI response is always structured like this 👇

📌 Question

📊 Market Snapshot

📰 Key News & Events

📈 Analysis & Explanation

🧠 Investor Takeaway

Clean, readable, and easy to understand.

▶️ How to Run This
1️⃣ Install dependencies
pip install yfinance feedparser gradio python-dotenv langchain faiss-cpu sentence-transformers

2️⃣ Set up environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here

3️⃣ Run the app
python app.py


Gradio will open the app in your browser 🚀

🖥️ UI Preview

Textbox to ask questions

Analyze button

Loading spinner with progress updates

Clean markdown output

Simple and student-friendly.

⚠️ Disclaimer

❌ Not financial advice

❌ Not guaranteed to be accurate

📚 Built for learning & experimentation only

Always do your own research before investing.

🚀 Why I Built This

To learn LLMs + finance

To experiment with RAG (Retrieval-Augmented Generation)

To understand Indian stock market data

To mix AI + real-world applications

#  ---------------------- TASK 3 ---------------------- 

📈 Hybrid Live Stock Prediction Engine (PyTorch + Yahoo Finance)

Hey!
This project is a live stock price prediction system built with PyTorch and Yahoo Finance data. It mixes long-term daily trends with short-term minute-by-minute movements using a hybrid LSTM model.

In simple words:
👉 One model looks at the big picture
👉 Another model looks at what’s happening right now
👉 A hybrid model combines both to predict the next minute

🚀 What This Project Does

📊 Downloads 2 years of daily stock data

⏱ Downloads last 6 days of 1-minute data

🧠 Trains two separate LSTM models

Daily LSTM → long-term trend

Minute LSTM → short-term momentum

🔗 Combines both using a hybrid model

🔮 Predicts:

Next 7 days using the daily model

Live 1-minute price using the hybrid model

🔁 Runs in a loop and updates every minute

🧩 Model Architecture (High Level)

Daily Model

Input: last 60 trading days

Output: next day close (normalized)

Minute Model

Input: last 60 minutes

Output: next minute close (normalized)

Hybrid Model

Uses pretrained LSTM layers from both models

Freezes them

Trains a small neural network on top

This keeps training fast and stable 👍

📂 Project Flow (Step by Step)

Download data

Daily (2 years, 1D)

Minute (last 6 days, 1m)

Normalize data

Daily and minute data are normalized separately

Create datasets

Sliding windows for time series learning

Train models

Train daily LSTM

Train minute LSTM

7-day forecast

Uses only the daily model

Train hybrid model

Daily + minute LSTMs frozen

Live prediction

Every minute:

Fetch new data

Predict next close

Update rolling window

Save minute data to CSV

⚙️ Configuration You Can Change

At the top of the file:

TICKER = "AAPL"        # Stock symbol
SEQ_LEN_MIN = 60       # Minutes used for prediction
SEQ_LEN_DAY = 60       # Days used for prediction
EPOCHS = 8             # Training epochs
BATCH_SIZE = 32


Want to predict a different stock?
Just change "AAPL" to something else like "MSFT" or "TSLA".

🖥️ Requirements

Make sure you have these installed:

pip install yfinance torch pandas numpy


Optional but recommended:

GPU with CUDA for faster training

📄 Output Files

minute_database.csv

Stores minute-by-minute data

Used as a rolling window for live prediction

⚠️ Important Notes

This is not financial advice

Yahoo Finance minute data can:

Have small delays

Occasionally skip minutes

The model predicts directional behavior, not guaranteed prices

Live predictions depend heavily on market conditions

🧪 Good Use Cases

Learning time-series forecasting

Practicing LSTM + PyTorch

Building a base for:

Alerts

Dashboards

Trading simulations

Portfolio / research projects

🔮 Possible Upgrades (Next Steps)

If you want to improve this later:

🔔 Add price-difference alerts

📊 Add a Streamlit dashboard

🧠 Add confidence or uncertainty estimation

⏱ Add alert cooldowns

🔁 Add automatic retraining

📉 Direction-only prediction logic

# ---------------------- PS2 ---------------------- 
#  ---------------------- Task 1 ---------------------- 

🇮🇳 Indian Stock Market AI Assistant

This is an AI-powered stock market assistant for Indian stocks 🇮🇳📊
You ask questions like “Why is Reliance falling today?” and it gives you a clean, structured analysis using:

live market data

latest news

AI reasoning (LLM + vector search)

Built mainly as a learning + experimentation project.

🤔 What does this thing do?

You type a normal human question like:

Why is TCS stock going down today?

Trend of Infosys share price

When will Reliance recover?

And it:

Understands your intent

Detects the Indian company

Fetches live market data

Pulls latest news

Explains everything in simple markdown format

🧠 Core Features

📊 Live Indian stock prices (BSE)

📰 Latest market news (Economic Times + Moneycontrol)

🤖 LLM-powered explanations

🔍 Vector search on news articles

🧭 Intent detection (why / when / trend)

🖥️ Simple Gradio web UI

⏳ Loading spinner + progress status (feels cool 😄)

⚙️ Tech Stack

Python 🐍

yFinance (market data)

LangChain

OpenAI-compatible LLM

HuggingFace Embeddings

FAISS (vector search)

Gradio (UI)

📂 How It Works (Simple Flow)

User asks a question

AI extracts:

intent

company name

BSE ticker

Fetches live or recent market data

Pulls latest financial news

Builds a vector database

Retrieves relevant context

AI generates a structured markdown explanation

🧪 Output Format

The AI response is always structured like this 👇

📌 Question

📊 Market Snapshot

📰 Key News & Events

📈 Analysis & Explanation

🧠 Investor Takeaway

Clean, readable, and easy to understand.

▶️ How to Run This
1️⃣ Install dependencies
pip install yfinance feedparser gradio python-dotenv langchain faiss-cpu sentence-transformers

2️⃣ Set up environment variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here

3️⃣ Run the app
python app.py


Gradio will open the app in your browser 🚀

🖥️ UI Preview

Textbox to ask questions

Analyze button

Loading spinner with progress updates

Clean markdown output

Simple and student-friendly.

⚠️ Disclaimer

❌ Not financial advice

❌ Not guaranteed to be accurate

📚 Built for learning & experimentation only

Always do your own research before investing.

🚀 Why I Built This

To learn LLMs + finance

To experiment with RAG (Retrieval-Augmented Generation)

To understand Indian stock market data

To mix AI + real-world applications

🛠️ Future Improvements

Add technical indicators (RSI, MACD, etc.)

Multi-stock comparison

Sentiment scoring

Historical trend charts

Deployment on cloud

