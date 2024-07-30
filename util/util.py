from PIL import Image
import base64
import requests
import json
import glob


# Function to convert image to base64  
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:  
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def getFileList(directory_path, file_extension):

    # Use glob to get a list of files (e.g. .mp4) in the directory  
    return glob.glob(directory_path + '/*' + file_extension)   

def call_llm_api(GPT4V_ENDPOINT, GPT4V_API_VERSION, GPT4V_KEY, encoded_image, encoded_image_text_pairs = None, which_config_enabled = None):  
    
    # Set parameters
    temperature = 0
    top_p = 0.95
    max_tokens = 500

    # Set version
    GPT4V_ENDPOINT = GPT4V_ENDPOINT + "?api-version=" + GPT4V_API_VERSION

    system_message_a = '''
    Take a deep breath.
    You are creating accurate text caption (25-30 words maximum) based on few-shot examples of images and their text given to you. 
    Do not output very short or 1-4 worded text caption. 
    If you don't see a person clearly, do not make assumption, just answer based on part of images that you can see clearly.
    
    Output format:
    {
        "caption": "Text caption goes here.", 
        "everyone_wearing_mask": "yes/no", 
        "anyone_wearing_glasses": "yes/no"
    }
    
    Just return {} object without saying json or anything else before it.
    '''
    
    system_message_b = '''
    Take a deep breath.
    You are creating accurate text answers based on few-shot examples of images and their text outputs given to you. 
    Do not output very short or 1-4 worded text caption. 
    Do not make assumptions when reviewing the image, just answer based on part of images that you can see clearly.
    
    Output format:
    {
        "question_1": "What are people doing?",
        "answer_1": "one word answer, long answer goes here.",
        
        "question_2": "Does this look like a photo taken indoors?",
        "answer_1": "one word answer, long answer goes here.",
        
        "question_3": "Is this photo taken during the day or night?",
        "answer_1": "one word answer, long answer goes here.",
        
        "question_4": "Are people carrying any items?",
        "answer_1": "one word answer, long answer goes here.",
    }
    
    Just return {} object without saying json or anything else before it.
    '''
    
    system_message = ''
    if which_config_enabled is None or which_config_enabled =='a':
        system_message = system_message_a
    elif which_config_enabled =='b':
        system_message = system_message_b
       

    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }
    
    payload = {}
    messages = []
    
    # Add system message
    messages.append(
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": system_message
            }
          ]
        }
    )
    
    # Add few-shot examples (if supplied), given an input image and expected text
    if encoded_image_text_pairs and len(encoded_image_text_pairs) > 0:
        for encoded_image_text_pair in encoded_image_text_pairs:
            
            # Example image
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f'data:image/jpeg;base64,{encoded_image_text_pair["base64_image"]}'
                            }
                        }
                    ]
                }
            )
            
            # Example response
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": encoded_image_text_pair["expected_response"]
                        }
                    ]
                }
            )
            
    # Add the image to generate annotation for
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    )
    
    
    # Assemble payload dictionary elements
    payload["messages"] = messages
    payload["temperature"] = temperature
    payload["top_p"] = top_p
    payload["max_tokens"] = max_tokens


    # Send request
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        # raise SystemExit(f"Failed to make the request. Error: {e}")
        print(f'LLM request failed: {e}')
        if response.json():
            print(f'response.json():{response.json()}')
        return 'request_error'

    return response.json()