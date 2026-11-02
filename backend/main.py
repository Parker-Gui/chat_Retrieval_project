from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import sys
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from typing import List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


env_path = os.path.join(os.path.dirname(__file__), ".env")
print(env_path)
# Load environment variables
load_dotenv(dotenv_path= env_path)

# Make the local `app` package importable regardless of the launch directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.rag.service import (  # noqa: E402
    PipelineConfigurationError,
    configured_system,
    get_pipeline,
    shutdown as shutdown_pipelines,
)
from app.rag.workflow import SUPPORTED_SYSTEMS  # noqa: E402


app = FastAPI(title="MovieGraph API", version="1.0.0")

# CORS configuration for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    # Optional: "pure_llm" | "vector_rag" | "kg_retrieval" | "hybrid".
    # Android only sends `message`, so the configured default applies.
    system: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str

class BloodPressureRecord(BaseModel):
    systolic: int
    diastolic: int
    heart_rate: Optional[int] = None
    timestamp: Optional[float] = None
    notes: Optional[str] = None

class BloodPressureAnalysisRequest(BaseModel):
    records: List[BloodPressureRecord]
    email: Optional[str] = None

class BloodPressureAnalysisResponse(BaseModel):
    analysis: str
    recommendations: List[str]
    alert_level: str
    email_sent: bool = False

# DeepSeek API configuration. Retrieval-only experiments do not require this key.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

@app.get("/")
async def root():
    return {"message": "MovieGraph API is running"}

@app.get("/health")
def health():
    """Report readiness without leaking configuration values."""
    try:
        system = configured_system()
    except PipelineConfigurationError as error:
        return {"status": "degraded", "detail": str(error)}
    return {
        "status": "ok",
        "system": system,
        "supported_systems": list(SUPPORTED_SYSTEMS),
        "api_key_configured": bool(DEEPSEEK_API_KEY),
    }


@app.on_event("shutdown")
def _shutdown_pipelines():
    """Release the shared Neo4j driver when the server stops."""
    shutdown_pipelines()


def _format_reply(state) -> str:
    """Combine the generated sentence with the complete answer list.

    Long MetaQA answer sets cannot fit in one sentence, so when the sentence does
    not mention every retrieved answer the full list is appended. The JSONL
    evaluation reads the structured fields directly and is unaffected by this.
    """
    text = (state.final_answer or "").strip()
    answers = [value for value in (state.answer_set or []) if value]
    if len(answers) > 1:
        lowered = text.casefold()
        if any(value.casefold() not in lowered for value in answers):
            text = f"{text}\n\nAnswers: " + ", ".join(answers)
    return text


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Answer a movie question through the configured retrieval pipeline.

    The pipeline is built once per process (embedding model and retrievers are
    reused), so requests do not pay the model-loading cost again. The Android
    contract stays `{"message": ...} -> {"reply": ...}`.
    """
    question = (request.message or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="message must not be empty")
    if request.system is not None and request.system not in SUPPORTED_SYSTEMS:
        raise HTTPException(
            status_code=400,
            detail=f"system must be one of {list(SUPPORTED_SYSTEMS)}",
        )
    try:
        pipeline = get_pipeline(request.system)
        state = pipeline.run(question)
        reply = _format_reply(state)
        if not reply:
            raise HTTPException(
                status_code=502,
                detail="The retrieval pipeline returned an empty answer",
            )
        return ChatResponse(reply=reply)
    except PipelineConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Answer generation failed: {type(error).__name__}: {error}",
        )

@app.post("/blood-pressure/analyze", response_model=BloodPressureAnalysisResponse)
async def analyze_blood_pressure(request: BloodPressureAnalysisRequest):
    """
    Analyze blood pressure data and provide AI recommendations
    """
    try:
        if not request.records:
            raise HTTPException(status_code=400, detail="No blood pressure records provided")
        
        # Analyze the data
        analysis_result = await analyze_bp_data(request.records)
        
        # Check for alerts
        alert_level = determine_alert_level(request.records)
        
        # Send email if needed
        email_sent = False
        if alert_level in ["high", "critical"] and request.email:
            email_sent = await send_alert_email(request.email, analysis_result, request.records)
        
        return BloodPressureAnalysisResponse(
            analysis=analysis_result["analysis"],
            recommendations=analysis_result["recommendations"],
            alert_level=alert_level,
            email_sent=email_sent
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

async def analyze_bp_data(records: List[BloodPressureRecord]) -> dict:
    """Use AI to analyze blood pressure data"""
    
    # Prepare data summary for AI
    recent_records = sorted(records, key=lambda x: x.timestamp or 0, reverse=True)[:10]
    avg_systolic = sum(r.systolic for r in recent_records) / len(recent_records)
    avg_diastolic = sum(r.diastolic for r in recent_records) / len(recent_records)
    
    data_summary = f"""
    最近{len(recent_records)}次血压记录：
    平均收缩压: {avg_systolic:.1f} mmHg
    平均舒张压: {avg_diastolic:.1f} mmHg
    
    最近记录详情：
    """

    
    for i, record in enumerate(recent_records[:5], 1):
        timestamp = datetime.fromtimestamp(record.timestamp or 0).strftime("%Y-%m-%d %H:%M") if record.timestamp else "未知时间"
        data_summary += f"{i}. {timestamp}: {record.systolic}/{record.diastolic} mmHg"
        if record.heart_rate:
            data_summary += f", 心率: {record.heart_rate} bpm"
        if record.notes:
            data_summary += f", 备注: {record.notes}"
        data_summary += "\n"

    # Create AI prompt for blood pressure analysis
    ai_prompt = f"""
    作为一位专业的健康顾问，请分析以下血压数据并提供建议：

    {data_summary}

    请从以下几个方面进行分析：
    1. 血压水平评估（正常、偏高、高血压等）
    2. 趋势分析（是否有改善或恶化趋势）
    3. 健康风险提示
    4. 具体的改善建议（饮食、运动、生活习惯等）
    5. 是否需要就医建议

    请用中文回答，语言要温和、专业，适合老年人理解。
    """

    # Call the DeepSeek API for analysis.
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="DEEPSEEK_API_KEY is not configured in backend/.env",
        )

    deepseek_request = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "user",
                "content": ai_prompt
            }
        ],
        "temperature": 0.3,  # Lower temperature for more consistent medical advice
        "max_tokens": 1500
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=deepseek_request
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"DeepSeek API error: {response.text}"
            )

        result = response.json()

    if "choices" in result and len(result["choices"]) > 0:
        ai_response = result["choices"][0]["message"]["content"]

        # Parse the response to extract analysis and recommendations
        analysis = ai_response
        recommendations = extract_recommendations(ai_response)

        return {
            "analysis": analysis,
            "recommendations": recommendations
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Invalid response format from DeepSeek API"
        )

def extract_recommendations(ai_response: str) -> List[str]:
    """Extract recommendations from AI response"""
    recommendations = []
    
    # Simple extraction - look for numbered lists or bullet points
    lines = ai_response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '•', '-', '建议', '推荐')):
            recommendations.append(line)
    
    return recommendations[:5]  # Limit to 5 recommendations

def determine_alert_level(records: List[BloodPressureRecord]) -> str:
    """Determine alert level based on blood pressure readings"""
    if not records:
        return "normal"
    
    recent_records = sorted(records, key=lambda x: x.timestamp or 0, reverse=True)[:3]
    
    for record in recent_records:
        # Critical levels
        if record.systolic >= 180 or record.diastolic >= 120:
            return "critical"
        # High levels
        if record.systolic >= 160 or record.diastolic >= 100:
            return "high"
        # Elevated levels
        if record.systolic >= 140 or record.diastolic >= 90:
            return "elevated"
    
    return "normal"

async def send_alert_email(email: str, analysis: dict, records: List[BloodPressureRecord]) -> bool:
    """Send alert email to family members"""
    try:
        if not EMAIL_USER or not EMAIL_PASSWORD:
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = email
        msg['Subject'] = "血压异常提醒 - 老人健康监测"
        
        # Email body
        recent_record = sorted(records, key=lambda x: x.timestamp or 0, reverse=True)[0]
        timestamp = datetime.fromtimestamp(recent_record.timestamp or 0).strftime("%Y-%m-%d %H:%M") if recent_record.timestamp else "未知时间"
        
        body = f"""
        尊敬的家庭成员，
        
        您的家人的血压监测系统检测到异常情况，请关注：
        
        最新血压记录：
        时间：{timestamp}
        血压：{recent_record.systolic}/{recent_record.diastolic} mmHg
        {"心率：" + str(recent_record.heart_rate) + " bpm" if recent_record.heart_rate else ""}
        
        AI分析结果：
        {analysis['analysis']}
        
        建议措施：
        {chr(10).join(analysis['recommendations']) if analysis['recommendations'] else "请及时关注血压变化"}
        
        建议：
        1. 密切关注血压变化
        2. 如有必要，请及时就医
        3. 保持健康的生活方式
        
        此邮件由AI血压监测系统自动发送。
        
        祝您和家人身体健康！
        """
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            # text = msg.as_string()
            server.send_message(msg)
        #
        # server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)

        # server.starttls()  # 启用TLS加密

        # server.login(EMAIL_USER, EMAIL_PASSWORD)
        # text = msg.as_string()
        #
        # server.sendmail(EMAIL_USER, email, text)
        # server.quit()
        
        return True
        
    except Exception as e:
        print(f"Email sending failed: {e}")
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
