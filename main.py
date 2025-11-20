from flask import Flask, request, jsonify
from functools import wraps
import io
import re
import time
import cloudscraper
import fitz
import pymupdf4llm
import google.generativeai as genai
from google.generativeai import types
import os

app = Flask(__name__)

# Configuration
API_KEY = os.environ.get('API_KEY', 'your-secret-api-key-here')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set!")
else:
    print(f"GEMINI_API_KEY is set")
    genai.configure(api_key=GEMINI_API_KEY)

print(f"API_KEY for authentication: {API_KEY}")

# API Key Authentication Decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({
                "error": "Missing API key",
                "message": "Please provide an API key in the Authorization header"
            }), 401
        
        if auth_header.startswith('Bearer '):
            provided_key = auth_header[7:]
        else:
            provided_key = auth_header
        
        if provided_key != API_KEY:
            return jsonify({
                "error": "Invalid API key",
                "message": "The provided API key is not valid"
            }), 403
        
        return f(*args, **kwargs)
    
    return decorated_function

# PDF Parsing Function
def parse_pdf_from_url(pdf_url: str, write_images: bool = False) -> dict:
    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(pdf_url, timeout=30)
        response.raise_for_status()

        pdf_stream = io.BytesIO(response.content)
        pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        
        pdf_doc.close()

        return {
            "status": "success",
            "text_content": text,
            "message": "Success"
        }
        
    except Exception as e:
        return {
            "status": "error1",
            "text_content": "",
            "message": str(e)
        }


# Tool function mapping
tool_functions = {"parse_pdf_from_url": parse_pdf_from_url}

# Define PDF parsing tool
pdf_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="parse_pdf_from_url",
            description="Downloads a PDF from a URL and converts it to plain text format. Use this when the user provides a PDF link.",
            parameters={
                "type": "object",
                "properties": {
                    "pdf_url": {
                        "type": "string",
                        "description": "The complete URL of the PDF file to download and parse"
                    },
                    "write_images": {
                        "type": "boolean",
                        "description": "Whether to extract and save images from the PDF (default: false)"
                    }
                },
                "required": ["pdf_url"]
            }
        )
    ]
)
#pdf_tool = Tool(function_declarations=[pdf_parsing_declaration])

def run_pdf_agent(user_prompt: str, system_prompt: str = None, max_iterations: int = 10):
    """
    Run Gemini agent with PDF parsing capability.
    """
    try:
        # Initialize model with corrected tool
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            tools=[pdf_tool],  # Pass as list
            system_instruction=system_prompt if system_prompt else "You are a helpful assistant."
        )
        
        chat = model.start_chat()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Iteration {iteration}")
            
            response = chat.send_message(user_prompt)
            
            # Check for function calls
            has_function_call = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    has_function_call = True
                    function_call = part.function_call
                    
                    fn_name = function_call.name
                    fn_args = dict(function_call.args)
                    
                    print(f"Calling function: {fn_name}")
                    print(f"Arguments: {fn_args}")
                    
                    # Execute the function
                    if fn_name in tool_functions:
                        result = tool_functions[fn_name](**fn_args)
                        print(f"Function result status: {result.get('status')}")
                        
                        # Send result back to model
                        response = chat.send_message(
                            genai.protos.Content(
                                parts=[
                                    genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=fn_name,
                                            response={'result': result}
                                        )
                                    )
                                ]
                            )
                        )
                        
                        # Return the final text response
                        return response.text
            
            # If no function call, return text
            if not has_function_call:
                return response.text
                
        return "Max iterations reached"
        
    except Exception as e:
        print(f"Error in run_pdf_agent: {e}")
        import traceback
        traceback.print_exc()
        raise

def parse_response(text: str):
    """
    Parse the agent response to extract subject, content, and URL.
    """
    subject_pattern = r'Subject:\s*(.+?)(?=\n\nContent:)'
    content_pattern = r'Content:\s*(.+)'

    subject_match = re.search(subject_pattern, text, re.DOTALL)
    content_match = re.search(content_pattern, text, re.DOTALL)

    subject = subject_match.group(1).strip() if subject_match else ""
    content = content_match.group(1).strip() if content_match else text  # Fallback to full text
    url = re.search(r'(https://[^\s]+)', text).group(1) if re.search(r'(https://[^\s]+)', text) else ""

    return {
        "time": f"Updated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        "subject": subject,
        "content": content,
        "url": url
    }

# API Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/parse-pdf', methods=['POST'])
@require_api_key
def parse_pdf():
    """
    Parse a PDF from URL and generate summary using Gemini.
    """
    try:
        data = request.get_json()
        print(f"\n--- Received Request ---")
        print(f"Data: {data}")
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_prompt = data.get('user_prompt')
        system_prompt = data.get('system_prompt', None)
        max_iterations = data.get('max_iterations', 10)
        
        print(f"User prompt: {user_prompt}")
        print(f"System prompt: {system_prompt}")
        
        if not user_prompt:
            return jsonify({"error": "user_prompt is required"}), 400
        
        # Run the agent
        print("Running PDF agent...")
        result = run_pdf_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_iterations=max_iterations
        )
        
        print(f"Agent result: {result[:200]}...")
        
        # Parse the response
        parsed = parse_response(result)
        print(f"Parsed response: {parsed}")
        
        return jsonify({
            "status": "success",
            "raw_response": result,
            "parsed_response": parsed
        }), 200
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "status": "error",
            "message": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/api/parse-pdf-direct', methods=['POST'])
@require_api_key
def parse_pdf_direct():
    """
    Direct PDF parsing without agent.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        pdf_url = data.get('pdf_url')
        write_images = data.get('write_images', False)
        
        if not pdf_url:
            return jsonify({"error": "pdf_url is required"}), 400
        
        result = parse_pdf_from_url(pdf_url, write_images)
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "message": "PDF Parser API",
        "endpoints": {
            "/health": "Health check",
            "/api/parse-pdf": "Parse PDF with Gemini agent",
            "/api/parse-pdf-direct": "Direct PDF parsing"
        },
        "authentication": "Required - Use Authorization header with API key"
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
