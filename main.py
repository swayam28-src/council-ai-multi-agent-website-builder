import os
import json
import httpx

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================================================
# MISTRAL API
# =========================================================

async def call_mistral(
    messages,
    model="mistral-small-latest",
    max_tokens=400
):
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY is missing. "
            "Please add it to your .env file."
        )

    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }

    async with httpx.AsyncClient(timeout=120.0) as client:

        response = await client.post(
            url,
            headers=headers,
            json=data
        )

        if response.status_code == 401:
            raise ValueError(
                "Mistral API key is unauthorized. "
                "Please check your MISTRAL_API_KEY."
            )

        if response.status_code == 403:
            raise ValueError(
                "Mistral API access was denied. "
                "Check your Mistral account/API permissions."
            )

        response.raise_for_status()

        result = response.json()

        return result["choices"][0]["message"]["content"]


# =========================================================
# COUNCIL AGENT LOOP
# =========================================================

MAX_REVISIONS = 3
TARGET_SCORE = 90


async def agent_loop(user_prompt: str):

    conversation_history = [
        {
            "role": "user",
            "content": f"User Request: {user_prompt}"
        }
    ]

    debate_sequence = [

        {
            "name": "Product Manager",
            "role": (
                "You are a Senior Product Manager. "
                "Propose the core features and user flow "
                "for the user's request. "
                "Keep it technical and concise, under 150 words."
            )
        },

        {
            "name": "UI/UX Designer",
            "role": (
                "You are a world-class UI/UX Designer. "
                "Based on the Product Manager proposal, "
                "describe the visual design, layout, colors, "
                "dark mode aesthetic and micro-interactions. "
                "Keep it under 150 words."
            )
        },

        {
            "name": "Security Agent",
            "role": (
                "You are a Senior Security Architect. "
                "Review the Product Manager and UI/UX proposals. "
                "Identify security vulnerabilities, "
                "input sanitization concerns and DOM edge cases. "
                "Keep it under 100 words."
            )
        },

        {
            "name": "The Interviewer",
            "role": (
                "You are a Devil's Advocate. "
                "Challenge the Product Manager, Designer and Security Agent. "
                "Point out unnecessary features and bloat. "
                "Demand practical pros and cons. "
                "Keep it under 150 words."
            )
        },

        {
            "name": "Product Manager (Rebuttal)",
            "role": (
                "You are the Product Manager. "
                "Defend essential features against the Interviewer "
                "and remove unnecessary features. "
                "Keep it under 100 words."
            )
        },

        {
            "name": "UI/UX Designer (Rebuttal)",
            "role": (
                "You are the UI/UX Designer. "
                "Simplify the layout while keeping the interface "
                "modern and high-end. "
                "Keep it under 100 words."
            )
        },

        {
            "name": "Implementation Plan Generator",
            "role": (
                "You are a Staff Software Architect. "
                "Synthesize the entire debate into a strict "
                "technical implementation plan for HTML, CSS and JavaScript. "
                "Keep it under 200 words."
            )
        }
    ]

    # =====================================================
    # DEBATE
    # =====================================================

    for step in debate_sequence:

        yield (
            "data: "
            + json.dumps({
                "agent": step["name"],
                "status": "typing"
            })
            + "\n\n"
        )

        messages = conversation_history.copy()

        messages.append({
            "role": "system",
            "content": step["role"]
        })

        try:

            reply = await call_mistral(
                messages,
                model="mistral-small-latest",
                max_tokens=350
            )

        except Exception as e:

            reply = f"Agent error: {str(e)}"

        conversation_history.append({
            "role": "assistant",
            "content": f"{step['name']}: {reply}"
        })

        yield (
            "data: "
            + json.dumps({
                "agent": step["name"],
                "message": reply
            })
            + "\n\n"
        )

    # =====================================================
    # CODING AGENT
    # =====================================================

    tester_prompt = """
You are a Frontend QA Tester.

Review the HTML generated by the Coding Agent.

Check for:

- Missing functions
- Broken event listeners
- Invalid selectors
- JavaScript syntax errors
- HTML syntax errors
- CSS problems
- Integration issues

Return ONLY valid JSON.

Use exactly this format:

{
    "score": 95,
    "issues": []
}

If there are issues, list them inside "issues".
"""

    current_code = ""
    best_code = ""
    best_score = 0

    tester_result = {
        "score": 0,
        "issues": []
    }

    for revision in range(MAX_REVISIONS):

        yield (
            "data: "
            + json.dumps({
                "agent": "Coding Agent",
                "status": "typing",
                "revision": revision + 1
            })
            + "\n\n"
        )

        # =================================================
        # FIRST CODING ROUND
        # =================================================

        if revision == 0:

            coding_prompt = """
You are a Senior Frontend Developer.

Using the complete council debate and implementation plan,
generate the requested web application.

Return ONE complete HTML document.

Requirements:

- Start with <!DOCTYPE html>
- Include HTML
- Include CSS inside <style>
- Include JavaScript inside <script>
- Implement the requested functionality
- Make the UI polished and responsive
- Do not explain the code
- Return ONLY the HTML
"""

            messages = conversation_history.copy()

            messages.append({
                "role": "system",
                "content": coding_prompt
            })

        # =================================================
        # FIXING ROUND
        # =================================================

        else:

            issues = tester_result.get("issues", [])

            fix_prompt = f"""
You are a Senior Frontend Developer.

Here is the current HTML:

{current_code}

The QA Tester found these issues:

{chr(10).join(issues)}

Fix the identified problems.

Return the COMPLETE corrected HTML.

Do not explain anything.
Return ONLY valid HTML beginning with <!DOCTYPE html>.
"""

            messages = [
                {
                    "role": "system",
                    "content": fix_prompt
                }
            ]

        try:

            current_code = await call_mistral(
                messages,
                model="mistral-small-latest",
                max_tokens=6500
            )

        except Exception as e:

            yield (
                "data: "
                + json.dumps({
                    "agent": "Coding Agent",
                    "message": f"Coding Agent error: {str(e)}"
                })
                + "\n\n"
            )

            break

        # Remove Markdown code fences if Mistral adds them

        current_code = (
            current_code
            .replace("```html", "")
            .replace("```HTML", "")
            .replace("```", "")
            .strip()
        )

        yield (
            "data: "
            + json.dumps({
                "agent": "Coding Agent",
                "code": current_code
            })
            + "\n\n"
        )

        # =================================================
        # TESTER
        # =================================================

        yield (
            "data: "
            + json.dumps({
                "agent": "Tester",
                "status": "typing"
            })
            + "\n\n"
        )

        tester_messages = [

            {
                "role": "system",
                "content": tester_prompt
            },

            {
                "role": "user",
                "content": current_code
            }
        ]

        try:

            tester_reply = await call_mistral(
                tester_messages,
                model="mistral-small-latest",
                max_tokens=300
            )

            tester_reply = (
                tester_reply
                .replace("```json", "")
                .replace("```JSON", "")
                .replace("```", "")
                .strip()
            )

            tester_result = json.loads(tester_reply)

        except Exception as e:

            tester_result = {
                "score": 0,
                "issues": [
                    f"Tester failed: {str(e)}"
                ]
            }

        score = tester_result.get("score", 0)

        if score > best_score:

            best_score = score
            best_code = current_code

        yield (
            "data: "
            + json.dumps({
                "agent": "Tester",
                "message": tester_result
            })
            + "\n\n"
        )

        if score >= TARGET_SCORE:
            break

    # =====================================================
    # FINAL RESULT
    # =====================================================

    yield (
        "data: "
        + json.dumps({
            "agent": "Final",
            "code": best_code,
            "score": best_score
        })
        + "\n\n"
    )

    yield "data: [DONE]\n\n"


# =========================================================
# API ROUTES
# =========================================================

@app.get("/api/stream")
async def stream_debate(prompt: str):

    return StreamingResponse(
        agent_loop(prompt),
        media_type="text/event-stream"
    )


@app.get("/", response_class=HTMLResponse)
async def read_index():

    with open(
        "static/index.html",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()
        