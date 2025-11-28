import json
import io
import re
import time
import cloudscraper
import fitz
import google.generativeai as genai
from google.generativeai import types
import os
import traceback

# Configuration
API_KEY = os.environ.get('API_KEY', 'secret-api-key')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set!")
else:
    print(f"GEMINI_API_KEY is set")
    genai.configure(api_key=GEMINI_API_KEY)

print(f"API_KEY for authentication: {API_KEY}")

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

def run_pdf_agent(user_prompt: str, system_prompt: str = None, max_iterations: int = 10):
    """Run Gemini agent with PDF parsing capability."""
    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            tools=[pdf_tool],
            system_instruction=system_prompt if system_prompt else "You are a helpful assistant."
        )
        generation_config = {"temperature": 0.1}
        
        chat = model.start_chat()
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Iteration {iteration}")
            
            response = chat.send_message(user_prompt, generation_config=generation_config)
            
            has_function_call = False
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    has_function_call = True
                    function_call = part.function_call
                    
                    fn_name = function_call.name
                    fn_args = dict(function_call.args)
                    
                    print(f"Calling function: {fn_name}")
                    print(f"Arguments: {fn_args}")
                    
                    if fn_name in tool_functions:
                        result = tool_functions[fn_name](**fn_args)
                        print(f"Function result status: {result.get('status')}")
                        
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
                        
                        for part in response.parts:
                            if part.text:
                                return part.text
                            elif part.function_call:
                                print(f"Model requested another function call: {part.function_call.name}")
            
            if not has_function_call:
                return response.text
            
        return "Max iterations reached"
        
    except Exception as e:
        print(f"Error in run_pdf_agent: {e}")
        traceback.print_exc()
        raise

def parse_response(text: str):
    """Parse the agent response to extract subject, content, and URL."""
    subject_pattern = r'Subject:\s*(.+?)(?=\n\nContent:)'
    content_pattern = r'Content:\s*(.+)'

    subject_match = re.search(subject_pattern, text, re.DOTALL)
    content_match = re.search(content_pattern, text, re.DOTALL)

    subject = subject_match.group(1).strip() if subject_match else ""
    content = content_match.group(1).strip() if content_match else text
    url = re.search(r'(https://[^\s]+)', text).group(1) if re.search(r'(https://[^\s]+)', text) else ""

    return {
        "time": f"Updated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        "subject": subject,
        "content": content,
        "url": url
    }

def lambda_handler(event, context):
    """Main Lambda handler - directly processes API Gateway events."""
    print(f"Received event: {json.dumps(event, indent=2)}")
    
    try:
        # Extract Authorization header
        auth_header = None
        if 'headers' in event:
            auth_header = event['headers'].get('Authorization') or event['headers'].get('authorization')
        
        # Validate API key
        if not auth_header:
            return {
                'statusCode': 401,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    "error": "Missing API key",
                    "message": "Please provide an API key in the Authorization header"
                })
            }
        
        if auth_header.startswith('Bearer '):
            provided_key = auth_header[7:]
        else:
            provided_key = auth_header
        
        if provided_key != API_KEY:
            return {
                'statusCode': 403,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid"
                })
            }
        
        # Get request body
        body = event.get('body', '{}')
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body
        
        print(f"Request data: {data}")
        
        if not data:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": "No JSON data provided"})
            }
        
        user_prompt = data.get('user_prompt')
        system_prompt = data.get('system_prompt', None)
        max_iterations = data.get('max_iterations', 10)
        
        if not user_prompt:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({"error": "user_prompt is required"})
            }
        
        print("Running PDF agent...")
        result = run_pdf_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_iterations=max_iterations
        )
        
        print(f"Agent result: {result}...")
        parsed = parse_response(result)
        print(f"Parsed response: {parsed}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                "status": "success",
                "raw_response": result,
                "parsed_response": parsed
            })
        }
        
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({"error": "Invalid JSON in request body"})
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                "status": "error",
                "message": str(e),
                "type": type(e).__name__
            })
        }


