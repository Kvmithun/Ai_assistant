import os
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
import logging

# ----------------------------------------------------
# 0. FUNCTION DEFINITIONS (Tool for Gemini)
# ----------------------------------------------------

# NOTE: In a real-world app, user_location would come from the frontend client.
# Here, we use a mock location for the demo.
MOCK_USER_LOCATION = {"lat": 40.7128, "lon": -74.0060}


def find_nearest_hospital(specialty: str, type: str, user_location: dict) -> str:
    """
    Looks up hospitals near the user's location based on medical specialty
    and hospital type (Government or Private). Returns a list of nearby
    hospitals and their details, including price range and whether subsidized
    care is available.
    """
    # This function uses the MOCK_USER_LOCATION for the demo

    if specialty.lower() == "pulmonology" and type.lower() == "government":
        return (
            "Found 2 hospitals near your mock location. **Govt. City Hospital** (4km, free care available) "
            "and **Dr. R.K. Clinic** (6km, General Practitioner, low cost). "
            "Please use Feature 1: Hospital Locator & Details for navigation and real-time availability."
        )
    if specialty.lower() == "cardiology":
        return "Found 1 specialist center: **Apollo Cardiac Unit** (8km, Private, High Price Range). Use Feature 2: Appointment Booking to check available slots."

    return "No specialized facilities found matching your criteria nearby. Consider consulting a General Practitioner."


# The list of tools to provide to the Gemini API
HOSPITAL_TOOLS = [find_nearest_hospital]

# ----------------------------------------------------
# 1. SETUP & CONFIGURATION
# ----------------------------------------------------

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- CRITICAL FIX FOR LOCAL TESTING ---
GEMINI_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyB3c4FG_uobnAkiSeFfxX9RfoJM70AZpwQ')  # Ensure this key is valid

try:
    client = genai.Client(api_key=GEMINI_KEY)

    # Initialize a chat session with the tool and model
    # Note: Using generate_content for better control over tool configuration

    # --- ENHANCED SYSTEM INSTRUCTION WITH STATUS TAGS AND MAPPING ---
    system_instruction = (
        "You are the Smart Health Connect AI, a compassionate and knowledgeable health triage assistant. "
        "Your primary goal is to provide preliminary, non-diagnostic guidance based on user symptoms. "

        # 1. STATUS TAG INSTRUCTION (CRITICAL FOR UI)
        "Always prefix your response with one of these exact tags to indicate the severity/scope: "
        "[TRIAGE] for emergencies or critical symptoms requiring immediate medical attention. "
        "[ADVICE] for non-emergency self-care, first-aid, or general health tips. "
        "[REFERRAL] for recommending a specialist, hospital type, or guiding the user to a specific app feature. "

        # 2. FEATURE MAPPING INSTRUCTION
        "When the user mentions **cost, affordability, or price comparison**, you MUST recommend **Feature 1: Hospital Locator & Details** to them. "
        "When the user asks to book a consult, recommend **Feature 2: Appointment Booking**. "
        "When the user asks about medicine information or finding a pharmacy, recommend **Feature 3: Pharmacy & Medicine Information**. "
        "When the user mentions severe financial need or donation, recommend **Feature 6: Community Support & Donations**. "

        "Keep responses concise and prioritize patient safety. DO NOT offer a diagnosis or replace a doctor."
    )
    AI_READY = True
except ValueError as e:
    app.logger.error(f"AI Initialization Error: {e}")
    AI_READY = False
    client = None


# ----------------------------------------------------
# 2. FLASK ROUTES
# ----------------------------------------------------

@app.route('/')
def index():
    """Renders the main chat interface page."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint to handle user messages and get Gemini's response."""
    if not AI_READY:
        return jsonify({
            'response': '[ADVICE] The AI service is currently unavailable. Please check the API key configuration on the server.'}), 503

    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Create the initial message content for the API call
        contents = [{'role': 'user', 'parts': [{'text': user_message}]}]

        # Send message with tools enabled
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=HOSPITAL_TOOLS
            )
        )

        # Handle Function Calling Loop
        if response.function_calls:
            function_calls = response.function_calls
            tool_responses = []

            for fc in function_calls:
                # Find the function implementation from the defined tools
                func_name = fc.name
                func_args = dict(fc.args)

                # Ensure the mock location is passed if the function needs it
                if 'user_location' in func_args and func_args['user_location'] is None:
                    func_args['user_location'] = MOCK_USER_LOCATION

                # Execute the function
                if func_name == 'find_nearest_hospital':
                    function_result = find_nearest_hospital(**func_args)

                    # Add the function response back to the chat history
                    tool_responses.append(types.Part.from_function_response(
                        name=func_name,
                        response={'content': function_result}
                    ))
                else:
                    tool_responses.append(types.Part.from_function_response(
                        name=func_name,
                        response={'content': 'Tool not found'}
                    ))

            # Send the tool output back to the model for the final text response
            contents.append({'role': 'model', 'parts': response.parts})  # Add previous model response
            contents.append({'role': 'tool', 'parts': tool_responses})  # Add tool output

            # Get the final natural language response
            final_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instruction)
            )
            return jsonify({'response': final_response.text})

        # If no function call, return the direct text response
        return jsonify({'response': response.text})

    except Exception as e:
        app.logger.error(f"Gemini API Runtime Error: {e}")
        # Ensure error response also has a status tag for the UI
        return jsonify({'error': '[ADVICE] A critical server error occurred while processing your request.'}), 500


# ----------------------------------------------------
# 3. RUN THE APP
# ----------------------------------------------------

if __name__ == '__main__':
    # Ensure directories exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    if not os.path.exists('static'):
        os.makedirs('static')

    # Note: If running locally, you must provide a valid API key above.
    app.run(debug=True)
