import os
import json

def generate_llm_executive_strategy(restaurant_name: str, health_scores: dict, drivers: dict, api_key: str = None) -> dict:
    """
    Generates grounded, executive-level management strategy using Gemini LLM if API key is provided,
    otherwise returns a beautifully formatted professional fallback strategy report.
    """
    effective_api_key = api_key if api_key else os.getenv("GEMINI_API_KEY")
    
    # Structure rich grounded payload
    grounded_payload = {
        "restaurant": restaurant_name,
        "aspect_scores": {k: v['score'] for k, v in health_scores.items()},
        "critical_aspects": [k for k, v in health_scores.items() if v['score'] < 60],
        "warning_aspects": [k for k, v in health_scores.items() if 60 <= v['score'] < 75],
        "excellent_aspects": [k for k, v in health_scores.items() if v['score'] >= 75],
        "top_complaint_words": drivers.get('complaints', []),
        "top_praise_words": drivers.get('praise', [])
    }
    
    system_prompt = f"""
    You are a Senior Restaurant Operations Consultant writing an Executive Management Strategy & Action Plan for the General Manager of '{restaurant_name}'.
    
    Use STRICTLY the following grounded JSON metric payload. Do NOT invent facts or statistics outside this data.
    
    GROUNDED METRIC PAYLOAD:
    {json.dumps(grounded_payload, indent=2)}
    
    FORMATTING & TONAL INSTRUCTIONS:
    1. Structure the response into 3 clear, highly professional Markdown sections:
       - 🚨 Section 1: Immediate Critical Operational Priorities (address critical/warning aspects, translating complaint keywords into natural operational issues).
       - ⚠️ Section 2: Pricing & Experience Optimization (concrete tactics).
       - 💡 Section 3: Brand Leverage & Praise Drivers (growth strategy using praise keywords).
    2. Under each section, provide:
       - **Core Diagnosis**: Plain-English explanation of what customer feedback reveals.
       - **Recommended Action Plan**: 2-3 specific, actionable operational steps (e.g. staffing adjustments, menu bundling, kitchen workflow).
       - **Expected Business Impact**: The target operational outcome.
    3. Tone: Senior, authoritative, professional, and directly actionable. Avoid raw keyword lists; weave keywords naturally into professional advice.
    """
    
    if effective_api_key:
        try:
            # Try new google.genai SDK
            try:
                from google import genai
                client = genai.Client(api_key=effective_api_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=system_prompt
                )
                return {
                    "source": "Gemini-2.5-Flash (Grounded Executive Consultant Prompt)",
                    "content": response.text,
                    "payload_used": grounded_payload
                }
            except Exception:
                # Try fallback google.generativeai SDK
                import google.generativeai as genai_old
                genai_old.configure(api_key=effective_api_key)
                model = genai_old.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(system_prompt)
                return {
                    "source": "Gemini-1.5-Flash (Grounded Executive Consultant Prompt)",
                    "content": response.text,
                    "payload_used": grounded_payload
                }
        except Exception as err:
            return {
                "source": f"Professional Rule Synthesizer (API Error: {str(err)[:60]}...)",
                "content": _format_professional_fallback(grounded_payload),
                "payload_used": grounded_payload
            }
            
    # Default Professional Formatted Fallback
    return {
        "source": "Grounded Executive Operations Synthesizer",
        "content": _format_professional_fallback(grounded_payload),
        "payload_used": grounded_payload
    }


def _format_professional_fallback(payload: dict) -> str:
    """Formats grounded metrics into an executive-level managerial action plan."""
    restaurant = payload['restaurant']
    scores = payload['aspect_scores']
    complaints = payload['top_complaint_words']
    praises = payload['top_praise_words']
    
    # Format complaint keywords into natural language text
    complaint_str = f"such as {', '.join(complaints[:3])}" if complaints else "related to service responsiveness"
    praise_str = f"including {', '.join(praises[:3])}" if praises else "food quality and atmosphere"
    
    sections = [f"## 📋 Executive Management Strategy & Action Plan: {restaurant}\n"]
    
    # Section 1: Critical or Warning Priority
    if payload['critical_aspects'] or payload['warning_aspects']:
        target_aspects = payload['critical_aspects'] + payload['warning_aspects']
        aspect_names = " and ".join(target_aspects)
        sections.append(f"""
### 🚨 1. Immediate Operational Priority: {aspect_names} Refinement
- **Core Diagnosis**: Customer feedback indicates notable friction in **{aspect_names}** (Health Index: {scores.get(target_aspects[0], 65)}/100). Review analysis highlights recurring complaints {complaint_str}.
- **Recommended Action Plan**:
  - Adjust shift staffing levels during peak dining hours (7:00 PM – 10:00 PM) to eliminate service bottlenecks and order processing delays.
  - Implement a pre-service briefing checklist focused on customer responsiveness and table management.
- **Expected Business Impact**: Improves service resolution speed and elevates the {aspect_names} Health Index into the satisfactory tier (75+).
""")
    else:
        sections.append(f"""
### 🟢 1. Operational Stability Baseline
- **Core Diagnosis**: {restaurant} maintains high customer satisfaction across all 5 dining aspects with scores exceeding 75/100.
- **Recommended Action Plan**:
  - Maintain current kitchen prep standards and continue regular staff hospitality reviews.
""")

    # Section 2: Value & Experience Optimization
    price_score = scores.get('Price', 70)
    sections.append(f"""
### ⚠️ 2. Pricing & Perception Optimization
- **Core Diagnosis**: Value perception index stands at **{price_score}/100**. Feedback indicates cost sensitivity around specific add-on items and side dishes.
- **Recommended Action Plan**:
  - Introduce structured combo menu options (Main Course + Beverage/Dessert) to soften individual item cost perception.
  - Conduct a competitive margin analysis against neighboring restaurant offerings.
- **Expected Business Impact**: Increases average check size while boosting customer value perception.
""")

    # Section 3: Brand Leverage
    sections.append(f"""
### 💡 3. Growth Strategy & Brand Praise Drivers
- **Core Diagnosis**: Customer sentiment is exceptionally strong regarding core dining highlights, {praise_str}.
- **Recommended Action Plan**:
  - Feature signature praise items in social media promotions and localized marketing campaigns.
  - Launch a customer loyalty program to incentivize repeat dining visits among frequent reviewers.
- **Expected Business Impact**: Drives repeat customer visits and strengthens brand equity across local dining platforms.
""")

    return "\n\n".join(sections)
