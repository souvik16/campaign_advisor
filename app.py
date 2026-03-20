"""
Campaign Advisor Web App
========================
Flask backend that exposes the Campaign Advisor logic as a REST API.
The frontend (index.html) calls /chat and /reset endpoints.
"""

import json
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────
client = OpenAI(api_key= "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")
MODEL = "gpt-4o-mini"

# ─────────────────────────────────────────────
# Session State (in-memory, single user)
# For multi-user production use Flask sessions or a DB
# ─────────────────────────────────────────────
INITIAL_STATE = {
    "objective": None,
    "program_type": None,
    "audience": None,
    "budget": None,
    "industry": None,
    "geography": None,
    "education_required": None
}

FIELD_PRIORITY = ["objective", "program_type", "audience", "budget", "industry", "geography"]

session_state = dict(INITIAL_STATE)
conversation_history = []
campaign_generated = False


def reset_session():
    global session_state, conversation_history, campaign_generated
    session_state = dict(INITIAL_STATE)
    conversation_history = []
    campaign_generated = False


# ─────────────────────────────────────────────
# Core Logic Functions
# ─────────────────────────────────────────────

def extract_state(user_input: str, current_state: dict) -> dict:
    prompt = f"""
You are a state extraction engine for a Campaign Advisor chatbot.

Your job: Parse the user's message and extract values for any of these fields:
- objective: The campaign goal (e.g., increase loyalty, drive trials, boost retention)
- program_type: Must be exactly one of:
    "Consumer Promotion" → promotions, offers, discounts, trial, sampling, contests, gifting
    "Loyalty Program"    → loyalty, rewards program, points, retention, repeat purchase
    "Hybrid"             → explicitly wants both types
    null                 → not clearly indicated
- audience: Target audience (e.g., retailers, end consumers, millennials, housewives)
- budget: Campaign budget (any format: rupees, dollars, or descriptive)
- industry: Business sector (e.g., FMCG, telecom, apparel, food & beverage)
- geography: Target region (e.g., pan-India, metro cities, Southeast Asia)
- education_required: Does campaign need consumer education? ("Yes" or "No")

Current known state:
{json.dumps(current_state, indent=2)}

User message: "{user_input}"

Rules:
1. Return ONLY a valid JSON object with the same 7 keys.
2. Keep existing values unless user explicitly provides new ones.
3. For program_type: only set if intent is clear and unambiguous.
4. For industry and geography: extract even if mentioned casually.
5. Do NOT add extra keys. No explanation. No markdown.

Return ONLY raw JSON.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        updated = json.loads(raw)
        for key in INITIAL_STATE:
            if key not in updated:
                updated[key] = current_state.get(key)
        return updated
    except json.JSONDecodeError:
        return current_state


def get_missing_fields(state: dict) -> list:
    return [f for f in FIELD_PRIORITY if not state.get(f)]


def generate_followup_question(state: dict, history: list) -> str:
    missing = get_missing_fields(state)
    if not missing:
        return ""

    next_field = missing[0]
    prior = [m["content"] for m in history if m["role"] == "assistant"]
    prior_summary = "\n".join(f"- {q}" for q in prior[-6:]) if prior else "None yet."

    prompt = f"""
You are a warm, experienced Campaign Advisor having a real conversation with a client.

Personality: thoughtful, curious, light humour, sounds like a real person — not a bot.

Current known information:
{json.dumps(state, indent=2)}

Questions already asked (DO NOT repeat phrasing or sentence structure):
{prior_summary}

Your task: Ask ONE question to find out "{next_field}".

Field guide:
- objective: Business outcome they want (drive trials, increase repeat purchase, build loyalty, boost awareness)
- program_type: "Consumer Promotion" (short-term offers, contests, sampling) vs "Loyalty Program" (points, tiers, long-term rewards) vs "Hybrid". Frame as a natural choice.
- audience: Retailers, distributors, end consumers, specific demographic?
- budget: Rough ballpark — any format (₹ lakhs, $, "modest", "flexible")
- industry: Business sector (FMCG, telecom, apparel, pharma, food & beverage)
- geography: Country, region, city tier where campaign runs

Rules:
1. Ask about ONLY "{next_field}".
2. Weave in known context to feel personal.
3. Vary sentence structure — never start same as prior questions.
4. Use natural transitions: "And...", "One thing I'd love to understand...", "Before I dig in...", "Quick one —", "Help me understand...", "Out of curiosity..."
5. For program_type: explain both options briefly in plain language.
6. Max 2-3 sentences. No bullets. No lists. Flowing prose only.
7. Occasionally add a brief remark showing you're thinking about their situation.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


def generate_campaign(state: dict) -> str:
    industry = state.get("industry")
    geography = state.get("geography")
    education_required = state.get("education_required") or "No"
    program_type = state.get("program_type")

    prompt = f"""
You are a senior Campaign Strategist specializing in channel loyalty and consumer promotions.

CLIENT BRIEF:
- Objective: {state['objective']}
- Program Type: {program_type}
- Target Audience: {state['audience']}
- Budget: {state['budget']}
- Industry: {industry}
- Geography: {geography}
- Consumer Education Required: {education_required}

The campaign must align strongly to the program type: "{program_type}".
- "Consumer Promotion": short-term incentives, trial drivers, offers, contests, gifting, sampling.
- "Loyalty Program": points, tiers, repeat engagement, long-term retention rewards.
- "Hybrid": blend short-term hooks with longer loyalty structure.

Generate a structured response with these sections:

🎯 RECOMMENDED CAMPAIGN STRATEGY

Campaign Type: [Channel Loyalty / Consumer Promotion / Hybrid]

Engagement Mechanics:
[How participants engage — points, tiers, instant wins, challenges, etc.]

Reward Strategy:
[What rewards are offered and how they are earned/redeemed]

Campaign Flow:
[Step-by-step journey from awareness → participation → reward]

Narrative Idea:
[1-2 sentence creative theme or tagline concept]

Justification:
[Why this approach fits the objective, audience, and budget]

---

💡 CAMPAIGN VARIATIONS

Generate exactly 3 variation ideas. For each:
Variation Name: [creative title]
Core Mechanic: [1 sentence]
Key Differentiator: [what makes it unique]
Best Suited When: [specific condition]

Keep tone expert but accessible. Be specific, not generic. Tailor everything to the brief.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


def answer_followup(user_input: str, state: dict) -> str:
    prompt = f"""
You are a Campaign Advisor consultant. You've already recommended a campaign strategy.

Client session context:
{json.dumps(state, indent=2)}

The client asks: "{user_input}"

Answer helpfully and concisely as a consultant. Reference campaign context where relevant.
Keep response under 150 words unless detail is truly needed.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    reset_session()
    opening = ("Hey there! I'm your Campaign Advisor — think of me as your strategic partner "
                "for building loyalty and promotion campaigns. So, what's the challenge you're "
                "trying to solve? Tell me what's on your mind and we'll figure out the best approach together.")
    conversation_history.append({"role": "assistant", "content": opening})
    return jsonify({"message": opening, "state": session_state})


@app.route("/chat", methods=["POST"])
def chat():
    global session_state, campaign_generated

    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    conversation_history.append({"role": "user", "content": user_input})

    # Step 1: Extract state
    session_state = extract_state(user_input, session_state)

    # Step 2: Check missing fields
    missing = get_missing_fields(session_state)

    if missing:
        reply = generate_followup_question(session_state, conversation_history)
        msg_type = "question"
        campaign_generated = False
    else:
        if not campaign_generated:
            bridge = "Perfect — I think I have a clear enough picture now. Give me a moment while I put together a strategy tailored to what you've shared..."
            campaign = generate_campaign(session_state)
            reply = bridge + "\n\n" + campaign
            msg_type = "campaign"
            campaign_generated = True
        else:
            keywords = ["change", "adjust", "different", "instead", "try", "what if", "modify", "update"]
            if any(kw in user_input.lower() for kw in keywords):
                session_state = extract_state(user_input, session_state)
                campaign = generate_campaign(session_state)
                reply = "Got it — let me revise the strategy based on your update...\n\n" + campaign
                msg_type = "campaign"
            else:
                reply = answer_followup(user_input, session_state)
                msg_type = "answer"

    conversation_history.append({"role": "assistant", "content": reply})

    return jsonify({
        "message": reply,
        "type": msg_type,
        "state": session_state,
        "missing_fields": missing
    })


@app.route("/state", methods=["GET"])
def get_state():
    return jsonify({"state": session_state, "missing": get_missing_fields(session_state)})


if __name__ == "__main__":
    reset_session()
    app.run(debug=True, port=5000)
