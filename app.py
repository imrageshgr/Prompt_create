from flask import Flask, render_template, request, jsonify
import anthropic
import os

app = Flask(__name__, template_folder='templetes')

# ─────────────────────────────────────────────
#  API Configuration  (opencode.ai / minimax)
# ─────────────────────────────────────────────
ANTHROPIC_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL", "https://opencode.ai/zen"
)
ANTHROPIC_API_KEY = os.environ.get(
    "ANTHROPIC_API_KEY",
    "sk-KAybtf9cxchHcc0YBnGpNPFRquluvsM8U3CM4QBTYLh0VPX7oJvCmVkpW6Vn5s6L",
)
ANTHROPIC_MODEL = os.environ.get(
    "ANTHROPIC_MODEL", "minimax-m2.5-free"
)

# Initialise the Anthropic client pointed at the custom endpoint
client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
)


# ─────────────────────────────────────────────
#  System prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert AI Prompt Engineer. "
    "Your job is to take raw, informal, or unclear user input and transform it into "
    "a clear, structured, and highly effective AI prompt. "
    "The output prompt must:\n"
    "- Be precise and unambiguous\n"
    "- Include necessary context and constraints\n"
    "- Be optimized for the best AI output\n"
    "- Use professional language\n"
    "- Be ready to paste directly into any AI tool (ChatGPT, Gemini, Claude, etc.)\n\n"
    "Return ONLY the refined prompt text. Do not add any explanations, preambles, "
    "or commentary — just the prompt itself."
)


# ─────────────────────────────────────────────
#  Core LLM call
# ─────────────────────────────────────────────
def call_llm_api(user_input: str) -> str:
    """
    Sends the user's raw idea to the LLM and returns a refined, optimised prompt.
    Uses the Anthropic Messages API via the opencode.ai endpoint.
    """
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_input},
        ],
    )
    # The model may return a ThinkingBlock before the TextBlock — find the first TextBlock
    for block in message.content:
        if hasattr(block, "text"):
            return block.text.strip()

    raise ValueError("No text content found in the model response.")


# ─────────────────────────────────────────────
#  Flask Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("main.html")


@app.route("/generate-prompt", methods=["POST"])
def generate_prompt():
    data = request.get_json()
    user_input = (data or {}).get("user_input", "").strip()

    if not user_input:
        return jsonify({"error": "Please enter some information first."}), 400

    try:
        refined_prompt = call_llm_api(user_input)
        return jsonify({"prompt": refined_prompt})

    except anthropic.APIStatusError as e:
        return jsonify({"error": f"API error {e.status_code}: {e.message}"}), 500

    except anthropic.APIConnectionError as e:
        return jsonify({"error": f"Connection error: {str(e)}"}), 503

    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)