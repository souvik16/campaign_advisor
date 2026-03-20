"""
Campaign Advisor Chatbot
========================
A conversational AI consultant for channel loyalty and consumer promotion strategies.
Uses OpenAI API with session-based state management for guided consultation flow.
"""

import json
import os
from openai import OpenAI

# ─────────────────────────────────────────────
# OpenAI Client Setup
# ─────────────────────────────────────────────
client = OpenAI(api_key="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")
MODEL = "gpt-4o-mini"

# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────
INITIAL_STATE = {
    "objective": None,       # e.g., "increase repeat purchases", "drive trial"
    "program_type": None,    # "Consumer Promotion" or "Loyalty Program" or "Hybrid"
    "audience": None,        # e.g., "retailers", "end consumers", "millennials"
    "budget": None,          # e.g., "₹10 lakhs", "$50,000", "low/medium/high"
    "industry": None,        # e.g., "FMCG", "telecom", "apparel"
    "geography": None,       # e.g., "pan-India", "Southeast Asia", "Tier-2 cities"
    "education_required": None  # Whether campaign needs consumer education: Yes/No
}

REQUIRED_FIELDS = ["objective", "program_type", "audience", "budget"]
# Priority order: objective first, then program_type (if not clear), then audience, budget
FIELD_PRIORITY = ["objective", "program_type", "audience", "budget"]


# ─────────────────────────────────────────────
# 1. STATE EXTRACTION
# ─────────────────────────────────────────────
def extract_state(user_input: str, current_state: dict) -> dict:
    """
    Use LLM to parse user input and extract/update state fields.
    Only overwrites a field if a new explicit value is provided.
    """
    prompt = f"""
You are a state extraction engine for a Campaign Advisor chatbot.

Your job: Parse the user's message and extract values for any of these fields:
- objective: The campaign goal (e.g., increase loyalty, drive trials, boost retention)
- program_type: The type of program the user wants. Must be exactly one of:
    "Consumer Promotion" → if user mentions promotions, offers, discounts, trial, sampling, contests, gifting to end consumers
    "Loyalty Program"    → if user mentions loyalty, rewards program, points, retention, repeat purchase scheme
    "Hybrid"             → if user explicitly wants both or mentions both types
    null                 → if the user has NOT clearly indicated either type (leave as null so we can ask)
- audience: Target audience (e.g., retailers, end consumers, millennials, housewives)
- budget: Campaign budget (any format: rupees, dollars, or descriptive like "moderate")
- industry: Business sector (e.g., FMCG, telecom, apparel, food & beverage)
- geography: Target region (e.g., pan-India, metro cities, Southeast Asia)
- education_required: Does the campaign need consumer education? ("Yes" or "No")

Current known state:
{json.dumps(current_state, indent=2)}

User message:
"{user_input}"

Rules:
1. Return ONLY a valid JSON object with the same 7 keys.
2. If a field is already known AND the user has not provided a new value for it, keep the existing value (do NOT set it to null).
3. If the user explicitly corrects a field, update it with the new value.
4. If a field is not mentioned or not clearly implied, keep its current value (null if it was null).
5. For program_type: only set it if the user's intent is clear and unambiguous. If in doubt, leave as null.
6. Do NOT add extra keys. Do NOT include explanation.

Return ONLY JSON, no markdown, no code blocks.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        updated = json.loads(raw)
        # Safety: ensure all expected keys exist
        for key in INITIAL_STATE:
            if key not in updated:
                updated[key] = current_state.get(key)
        return updated
    except json.JSONDecodeError:
        # Fallback: return current state unchanged if parsing fails
        print("[DEBUG] State extraction failed to parse JSON. Keeping current state.")
        return current_state


# ─────────────────────────────────────────────
# 2. MISSING FIELD DETECTION
# ─────────────────────────────────────────────
def get_missing_fields(state: dict) -> list:
    """Return list of required fields that are still null/empty."""
    return [field for field in FIELD_PRIORITY if not state.get(field)]


# ─────────────────────────────────────────────
# 3. FOLLOW-UP QUESTION GENERATION
# ─────────────────────────────────────────────
def generate_followup_question(state: dict) -> str:
    """
    Generate a single, natural follow-up question for the highest-priority missing field.
    """
    missing = get_missing_fields(state)
    if not missing:
        return ""

    next_field = missing[0]

    prompt = f"""
You are a friendly, experienced Campaign Advisor consultant.

You are gathering information to build a marketing campaign strategy.

Current known information:
{json.dumps(state, indent=2)}

The next piece of information you need is: "{next_field}"

Field descriptions:
- objective: What the campaign should achieve (e.g., loyalty, trial, retention, awareness)
- program_type: Whether the client wants a "Consumer Promotion" (offers, contests, sampling, discounts for end consumers)
  or a "Loyalty Program" (points, tiers, rewards for repeat behavior) or a "Hybrid" of both.
  Ask this in a warm, options-based way — give them the two clear choices.
- audience: Who the campaign targets (e.g., retailers, distributors, end consumers, demographics)
- budget: Approximate budget available for the campaign

Rules:
1. Ask ONLY ONE question about "{next_field}".
2. Keep it conversational and natural, like a consultant would ask.
3. Reference what you already know to make the question feel contextual.
4. For program_type specifically: present it as a clear choice between Consumer Promotion vs Loyalty Program
   (briefly explain each in one phrase so the user can decide easily). You may mention Hybrid as an option too.
5. Do NOT ask about any other field.
6. Keep it to 2-3 sentences max.
7. Do NOT use bullet points or lists — write it as flowing natural speech.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 4. CAMPAIGN GENERATION
# ─────────────────────────────────────────────
def generate_campaign(state: dict) -> str:
    """
    Generate a full campaign recommendation with 2-3 variations
    once all required fields are available.
    """
    # Apply smart defaults for optional fields
    industry = state.get("industry") or "FMCG"
    geography = state.get("geography") or "pan-India"
    education_required = state.get("education_required") or "No"
    program_type = state.get("program_type") or "Consumer Promotion"

    prompt = f"""
You are a senior Campaign Strategist specializing in channel loyalty and consumer promotions.

Based on the following client brief, generate a detailed campaign recommendation.

CLIENT BRIEF:
- Objective: {state['objective']}
- Program Type: {program_type}
- Target Audience: {state['audience']}
- Budget: {state['budget']}
- Industry: {industry} {"(inferred)" if not state.get("industry") else ""}
- Geography: {geography} {"(inferred)" if not state.get("geography") else ""}
- Consumer Education Required: {education_required} {"(inferred)" if not state.get("education_required") else ""}

The campaign must be strongly aligned to the program type: "{program_type}".
- If "Consumer Promotion": focus on short-term incentives, trial drivers, offers, contests, gifting, sampling.
- If "Loyalty Program": focus on points, tiers, repeat engagement mechanics, long-term retention rewards.
- If "Hybrid": blend both — short-term promotion hooks that feed into a longer loyalty structure.

Generate a structured response with the following sections:

═══════════════════════════════════════════
🎯 RECOMMENDED CAMPAIGN STRATEGY
═══════════════════════════════════════════

**Campaign Type:** [Channel Loyalty / Consumer Promotion / Hybrid]

**Engagement Mechanics:**
[How participants engage — points, tiers, instant wins, challenges, etc.]

**Reward Strategy:**
[What rewards are offered and how they are earned/redeemed]

**Campaign Flow:**
[Step-by-step journey from awareness → participation → reward]

**Narrative Idea:**
[1-2 sentence creative theme or tagline concept]

**Justification:**
[Why this approach fits the objective, audience, and budget]

───────────────────────────────────────────
💡 CAMPAIGN VARIATIONS
───────────────────────────────────────────

Generate exactly 3 campaign variation ideas. For each:
- Variation Name (creative title)
- Core Mechanic (1 sentence)
- Key Differentiator (what makes it unique)
- Best Suited When (specific condition)

Keep the tone expert but accessible. Avoid jargon overload.
Be specific with mechanics, not generic. Tailor everything to the brief.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 5. DISPLAY HELPERS
# ─────────────────────────────────────────────
def print_advisor(message: str):
    """Print advisor response with formatting."""
    print(f"\n🤝 Advisor: {message}\n")


def print_user_echo(message: str):
    """Optional: echo user input back cleanly (omitted for cleaner CLI)."""
    pass


def print_divider():
    print("─" * 60)


def print_state_debug(state: dict):
    """Print current state for transparency (can be toggled off)."""
    filled = {k: v for k, v in state.items() if v}
    if filled:
        print(f"  📋 [State so far: {', '.join(f'{k}={v}' for k, v in filled.items())}]")


# ─────────────────────────────────────────────
# 6. MAIN CHAT LOOP
# ─────────────────────────────────────────────
def main_chat_loop():
    """
    Primary conversation loop. Runs until user types 'exit'.
    Manages state across turns and drives the consultant flow.
    """
    print("\n" + "═" * 60)
    print("  🎯  CAMPAIGN ADVISOR  |  Loyalty & Promotions Consultant")
    print("═" * 60)
    print("  Type 'exit' to end the session.")
    print("  Type 'state' to see current session info.")
    print("  Type 'restart' to start over.")
    print("─" * 60 + "\n")

    state = dict(INITIAL_STATE)
    campaign_generated = False

    # Opening message
    print_advisor(
        "Hello! I'm your Campaign Advisor. I help brands design powerful channel loyalty "
        "and consumer promotion strategies.\n\n"
        "   Tell me about your campaign — what are you hoping to achieve? "
        "You can share as much or as little as you'd like to start."
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Session ended. Good luck with your campaign!\n")
            break

        if not user_input:
            continue

        # ── Special commands ──────────────────────────────────
        if user_input.lower() == "exit":
            print("\n👋 Thanks for the session! Best of luck with your campaign.\n")
            break

        if user_input.lower() == "restart":
            state = dict(INITIAL_STATE)
            campaign_generated = False
            print_advisor(
                "Sure, let's start fresh! What campaign challenge can I help you with today?"
            )
            continue

        if user_input.lower() == "state":
            print("\n📋 Current Session State:")
            print(json.dumps(state, indent=2))
            print()
            continue

        # ── Step 1: Extract / update state ───────────────────
        state = extract_state(user_input, state)
        print_state_debug(state)

        # ── Step 2: Check required fields ────────────────────
        missing = get_missing_fields(state)

        if missing:
            # ── Step 3a: Ask follow-up for next missing field ─
            question = generate_followup_question(state)
            print_advisor(question)
            campaign_generated = False  # Reset in case user changed something

        else:
            # ── Step 3b: Generate campaign recommendation ─────
            if not campaign_generated:
                print_advisor(
                    "Great, I have everything I need! Let me put together a campaign strategy for you..."
                )
                campaign = generate_campaign(state)
                print("\n" + campaign + "\n")
                print_divider()
                print_advisor(
                    "That's my full recommendation! Would you like to:\n"
                    "   • Explore a specific variation in more detail?\n"
                    "   • Adjust any campaign parameter (audience, budget, etc.)?\n"
                    "   • Or type 'restart' to plan a completely new campaign."
                )
                campaign_generated = True
            else:
                # User is following up after campaign was generated
                # Re-extract state in case they want adjustments
                missing_now = get_missing_fields(state)
                if not missing_now:
                    # Regenerate if they seem to be tweaking
                    keywords = ["change", "adjust", "different", "instead", "try", "what if", "modify", "update"]
                    if any(kw in user_input.lower() for kw in keywords):
                        state = extract_state(user_input, state)
                        print_advisor(
                            "Got it — let me revise the strategy based on your update..."
                        )
                        campaign = generate_campaign(state)
                        print("\n" + campaign + "\n")
                        print_divider()
                        print_advisor(
                            "Updated strategy above! Feel free to keep refining or type 'restart' for a new campaign."
                        )
                    else:
                        # General follow-up question — answer it in context
                        answer = answer_followup(user_input, state)
                        print_advisor(answer)


# ─────────────────────────────────────────────
# 7. CONTEXTUAL FOLLOW-UP ANSWERING
# ─────────────────────────────────────────────
def answer_followup(user_input: str, state: dict) -> str:
    """
    After a campaign is generated, handle follow-up questions
    in the context of the current campaign state.
    """
    prompt = f"""
You are a Campaign Advisor consultant. You've already recommended a campaign strategy.

Client's session context:
{json.dumps(state, indent=2)}

The client is now asking a follow-up question:
"{user_input}"

Answer helpfully and concisely as a consultant would. 
Reference the campaign context where relevant.
Keep response under 150 words unless detail is truly needed.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main_chat_loop()
