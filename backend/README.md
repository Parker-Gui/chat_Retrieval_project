# Chatbox Backend Setup Instructions

## Prerequisites
1. Python 3.11
2. DeepSeek API Key (only required for answer generation, not Step 5 retrieval)

## Setup Steps

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables
1. Copy `env.example` to `.env`
2. Get your API key from [DeepSeek Platform](https://platform.deepseek.com/api_keys)
3. Replace `your_api_key_here` with your actual API key

### 3. Run the Server
```bash
python main.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### POST /chat
Send a chat message and get AI response

**Request:**
```json
{
    "message": "Hello, how are you?"
}
```

**Response:**
```json
{
    "reply": "Hello! I'm doing well, thank you for asking. How can I help you today?"
}
```

## DeepSeek configuration

The default is `deepseek-flash`. Change `DEEPSEEK_MODEL` in `.env` to use a
different model supported by your DeepSeek account.

## Testing
You can test the API using curl:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello!"}'
```
