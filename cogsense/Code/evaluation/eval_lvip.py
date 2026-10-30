import os
import json
import base64
import re
import io
import argparse
import random
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import sys
import torch
from transformers import AutoProcessor
from datetime import datetime

# Import custom Qwen3-VL model with LVIP Head support
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qwen-vl-finetune"))
from qwenvl.models.modeling_cogsense import CogSenseForConditionalGeneration as Qwen3VLForConditionalGeneration


# ----------------------------
# Configuration
# ----------------------------
load_dotenv()

SYSTEM_PROMPT = "You are a visual reasoning assistant. Analyze the pattern in the given examples and select the correct answer from the provided options. You MUST output the final answer first."
IMAGE_ROOT = "./data/CogSense-Bench"  # default; overridden by --image_root

# ----------------------------
# Helper functions
# ----------------------------
def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def combine_images_horizontal(image_paths, labels=None, spacing=20, label_height=30):
    """
    Combine multiple images horizontally into one image
    
    Args:
        image_paths: List of image file paths
        labels: List of labels (e.g., ['A', 'B', 'C', 'D'])
        spacing: Horizontal spacing between images in pixels
        label_height: Height reserved for labels above images
    
    Returns:
        PIL Image object
    """
    images = [Image.open(path) for path in image_paths]
    
    # Calculate dimensions
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
    
    # Add space for labels if provided
    canvas_height = max_height + (label_height if labels else 0)
    
    # Create canvas
    combined = Image.new('RGB', (total_width, canvas_height), 'white')
    
    # Paste images
    x_offset = 0
    for i, img in enumerate(images):
        # Calculate vertical position to center image
        y_offset = label_height if labels else 0
        y_offset += (max_height - img.height) // 2
        
        # Add label if provided
        if labels and i < len(labels):
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined)
            label_text = labels[i]
            # Use default font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw label above image
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x_offset + (img.width - text_width) // 2
            draw.text((text_x, 5), label_text, fill='black', font=font)
        
        combined.paste(img, (x_offset, y_offset))
        x_offset += img.width + spacing
    
    return combined


def pil_image_to_base64(pil_image):
    """
    Convert PIL Image to base64 string
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def extract_answer_with_llm(model_response, openai_client, num_options):
    """
    Use GPT-4o to extract the answer from model response
    
    Args:
        model_response: The raw response from the model
        openai_client: OpenAI client instance
        num_options: Number of available options
    
    Returns:
        (answer_letter, extraction_success)
        - answer_letter: Extracted answer letter or None if extraction fails
        - extraction_success: Whether extraction was successful
    """
    # Generate option letters based on num_options
    option_letters = [chr(ord('A') + i) for i in range(num_options)]
    options_str = ", ".join(option_letters)
    
    extraction_prompt = f"""You are an answer extraction assistant. Given a model's response to a multiple-choice question, extract only the final answer option letter.

The available options are: {options_str}

Model response:
{model_response}

Instructions:
- Extract ONLY the final answer letter (e.g., {', '.join(option_letters[:min(4, num_options)])})
- If the response contains multiple mentions, return the final answer
- Return ONLY the letter, nothing else
- If no clear answer can be found, return "NONE"

Answer:"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        extracted = response.choices[0].message.content.strip().upper()
        
        # Validate the extracted answer
        if extracted in option_letters:
            return extracted, True
        elif extracted == "NONE":
            return None, False
        else:
            # Try to extract from the response
            for opt in option_letters:
                if opt in extracted:
                    return opt, True
            return None, False
    
    except Exception as e:
        print(f"Error during LLM extraction: {e}")
        return None, False


def extract_answer(response_text, num_options):
    """
    Extract answer letter from model response using rule-based method
    
    Priority:
    1. <answer>X</answer> tag format
    2. "答案是X" / "Answer: X" / "answer is X" format
    3. Single uppercase letter (A/B/C/D, etc.)
    
    Args:
        response_text: Model response text
        num_options: Number of options (used to validate answer validity)
    
    Returns:
        (answer_letter, extraction_success)
        - answer_letter: Extracted answer letter (A/B/C/D, etc.), returns None if failed
        - extraction_success: Whether extraction was successful
    """
    if not response_text:
        return None, False
    
    # 1. Try to extract <answer>X</answer> format
    match = re.search(r'<answer>([A-Z])</answer>', response_text, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if ord(answer) < ord('A') + num_options:
            return answer, True
    
    # 2. Try to extract "答案是X" / "Answer: X" / "answer is X" format
    match = re.search(r'(?:答案是|Answer:\s*|answer is\s*)([A-Z])', response_text, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        if ord(answer) < ord('A') + num_options:
            return answer, True
    
    # 3. Find all standalone uppercase letters
    letters = re.findall(r'\b([A-Z])\b', response_text)
    valid_letters = [l for l in letters if ord(l) < ord('A') + num_options]
    if valid_letters:
        # Take the first valid letter
        return valid_letters[0], True
    
    return None, False


def verify_answer(extracted_answer, ground_truth):
    """
    Verify if the extracted answer is correct
    
    Args:
        extracted_answer: Extracted answer letter
        ground_truth: Ground truth answer (extracted from solution field)
    
    Returns:
        bool: Whether it is correct
    """
    if extracted_answer is None:
        return False
    
    # Extract answer letter from ground_truth (format: "<answer>A</answer>")
    gt_letter, _ = extract_answer(ground_truth, num_options=26)  # Maximum 26 options
    
    return extracted_answer == gt_letter


# ----------------------------
# Chat input construction
# ----------------------------
def make_chat_input_arc_agi_single_choice(entry, image_root=IMAGE_ROOT, combine_options=False):
    """
    Create single-choice input for ARC-AGI problems (new format)
    
    Display:
    1. Pattern image sequences (before/after pairs)
    2. Question text
    3. Option images (A/B/C/D)
    
    Args:
        entry: Data entry
        image_root: Image root directory
        combine_options: If True, combine all option images into one image
    
    Requires model to select answer from options
    """
    messages = []
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    
    user_content = []
    
    # Add problem description
    user_content.append({
        "type": "text",
        "text": f"{entry['problem']}\n\n"
    })
    
    # 1. Add pattern images (before/after pairs)
    num_patterns = len(entry.get("pattern", []))
    if num_patterns > 0:
        user_content.append({
            "type": "text",
            "text": f"Pattern examples ({num_patterns} before-after pairs):\n"
        })
        
        for k, pattern_item in enumerate(entry["pattern"]):
            user_content.append({
                "type": "text",
                "text": f"\nExample {k+1} - Before:"
            })
            before_path = os.path.join(image_root, pattern_item['before_image_path'])
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_to_base64(before_path)}"
                }
            })
            user_content.append({
                "type": "text",
                "text": "After:"
            })
            after_path = os.path.join(image_root, pattern_item['after_image_path'])
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_to_base64(after_path)}"
                }
            })
    
    # 2. Add option images
    num_options = len(entry.get("options", []))
    option_letters = [chr(ord('A') + i) for i in range(num_options)]
    
    if combine_options:
        # Combine all options into one image
        user_content.append({
            "type": "text",
            "text": f"\n\nAnswer choices ({'/'.join(option_letters)}):"
        })
        
        option_paths = [os.path.join(image_root, opt) for opt in entry['options']]
        combined_image = combine_images_horizontal(option_paths, labels=option_letters)
        combined_base64 = pil_image_to_base64(combined_image)
        
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{combined_base64}"
            }
        })
    else:
        # Add options separately
        user_content.append({
            "type": "text",
            "text": "\n\nAnswer choices:"
        })
        
        for i, option_path in enumerate(entry['options']):
            label = option_letters[i]
            user_content.append({
                "type": "text",
                "text": f"\n{label}:"
            })
            full_path = os.path.join(image_root, option_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_to_base64(full_path)}"
                }
            })
    
    # 3. Add answer instructions
    user_content.append({
        "type": "text",
        "text": f"\n\nPlease select the correct answer from options {'/'.join(option_letters)}.\n"
               "Provide your answer in the format: <answer>X</answer> where X is the option letter."
    })
    
    messages.append({"role": "user", "content": user_content})
    return messages


def make_chat_input_multiple_choice(entry, image_root=IMAGE_ROOT, combine_options=False):
    """
    Multiple choice format (RAVEN, PGM, KiVA): has options field
    
    Args:
        entry: Data entry
        image_root: Image root directory
        combine_options: If True, combine all option images into one image
    """
    messages = []
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    
    user_content = []
    user_content.append({
        "type": "text",
        "text": f"{entry['problem']}\n\nQuestion image(s):\n"
    })
    
    # Process question images (could be single or multiple)
    path = entry['path']
    if isinstance(path, str):
        path = [path]
    
    for i, img_path in enumerate(path):
        if len(path) > 1:
            user_content.append({
                "type": "text",
                "text": f"\nQuestion image {i+1}:"
            })
        full_img_path = os.path.join(image_root, img_path)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_to_base64(full_img_path)}"
            }
        })
    
    # Add options
    num_options = len(entry['options'])
    option_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    option_letters = [option_labels[i] if i < len(option_labels) else str(i) for i in range(num_options)]
    
    if combine_options:
        # Combine all options into one image
        user_content.append({
            "type": "text",
            "text": f"\n\nChoices ({'/'.join(option_letters)}):"
        })
        
        option_paths = [os.path.join(image_root, opt) for opt in entry['options']]
        combined_image = combine_images_horizontal(option_paths, labels=option_letters)
        combined_base64 = pil_image_to_base64(combined_image)
        
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{combined_base64}"
            }
        })
    else:
        # Add options separately
        user_content.append({
            "type": "text",
            "text": "\n\nChoices:"
        })
        
        for i, option_path in enumerate(entry['options']):
            label = option_letters[i]
            user_content.append({
                "type": "text",
                "text": f"\n{label}:"
            })
            full_option_path = os.path.join(image_root, option_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image_to_base64(full_option_path)}"
                }
            })
    
    user_content.append({
        "type": "text",
        "text": "\n\nPlease respond with ONLY the letter of your answer (e.g., 'A', 'B', 'C', etc.)."
    })
    
    messages.append({"role": "user", "content": user_content})
    return messages


def make_chat_input(entry, image_root=IMAGE_ROOT, model_key=None):
    """
    Unified entry function, select appropriate handler based on data fields
    
    Args:
        entry: Data entry
        image_root: Image root directory
        model_key: Model identifier (e.g., "internvl3-78b")
    """
    # Determine if we should combine options based on model type
    combine_options = False
    if model_key and "intern" in model_key.lower():
        combine_options = True
    
    if "pattern" in entry:
        # ARC-AGI type: Show pattern images, but require selecting answer from options
        return make_chat_input_arc_agi_single_choice(entry, image_root, combine_options=combine_options)
    elif "options" in entry:
        # Other types (Bongard, RAVEN, CVR, STARE, etc.): Standard multiple choice
        return make_chat_input_multiple_choice(entry, image_root, combine_options=combine_options)
    else:
        raise ValueError(f"Unknown dataset format for entry: {entry.get('problem_id', 'unknown')}")


# ----------------------------
# Model inference function
# ----------------------------
def load_model(model_path, device):
    """Load Qwen3-VL model and processor"""
    print(f"Loading model from {model_path} on device {device}...")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)

    # Check if model has LVIP Head
    if hasattr(model, 'model') and hasattr(model.model, 'nfp_head') and model.model.nfp_head is not None:
        print("✓ Model has LVIP Head (will not be used during inference)")
    
    print(f"Model loaded successfully on {device}")
    return model, processor


def call_model(model, processor, messages):
    """
    Call local Hugging Face model for inference
    
    Args:
        model: Loaded model
        processor: Loaded processor
        messages: Message list
    
    Returns:
        Model response text
    """
    # Convert OpenAI-style messages to Qwen3-VL format
    qwen_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            # Prepend system prompt to first user message (Qwen3-VL doesn't have separate system role)
            continue
        elif msg["role"] == "user":
            content = msg["content"]
            qwen_content = []
            
            # Add system prompt at the beginning of user content if available
            system_prompt = None
            for m in messages:
                if m["role"] == "system":
                    system_prompt = m["content"]
                    break
            
            if system_prompt and len(qwen_messages) == 0:
                qwen_content.append({
                    "type": "text",
                    "text": f"{system_prompt}\n\n"
                })
            
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        qwen_content.append({
                            "type": "text",
                            "text": item["text"]
                        })
                    elif item["type"] == "image_url":
                        # Convert base64 to PIL Image
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            base64_data = image_url.split(",")[1]
                            image_bytes = base64.b64decode(base64_data)
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            qwen_content.append({
                                "type": "image",
                                "image": pil_image
                            })
            else:
                qwen_content.append({
                    "type": "text",
                    "text": content
                })
            
            qwen_messages.append({
                "role": "user",
                "content": qwen_content
            })
    
    # Preparation for inference
    inputs = processor.apply_chat_template(
        qwen_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.1)
    
    # Trim input tokens from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()


# ----------------------------
# Main distributed evaluation
# ----------------------------
def main():
    """
    Distributed evaluation function
    """
    parser = argparse.ArgumentParser(description='Evaluation on single GPU')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Output directory for results andreports')
    parser.add_argument('--bench_path', type=str, default=os.path.join(IMAGE_ROOT, 'CogSense-Bench.jsonl'),
                        help='Path to CogSense-Bench.jsonl')
    parser.add_argument('--image_root', type=str, default=None,
                        help='Root directory of the benchmark images (default: directory of --bench_path)')
    parser.add_argument('--use_llm_extraction', action='store_true',
                        help='Use LLM (GPT-4o-mini) to extract answers instead of rule-based extraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling data (default: 42)')
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Do not shuffle data before evaluation')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with a small subset of data for quick debugging')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device {device}")
    
    # Initialize OpenAI client if using LLM extraction
    openai_client = None
    if args.use_llm_extraction:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment. Falling back to rule-based extraction.")
            args.use_llm_extraction = False
        else:
            base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            openai_client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"Using LLM extraction (GPT-4o-mini)")
    
    # Load test data
    test_jsonl_path = args.bench_path
    image_root = args.image_root or os.path.dirname(os.path.abspath(test_jsonl_path))
    
    if not os.path.exists(test_jsonl_path):
        print(f"Error: Test set file not found: {test_jsonl_path}")
        return
    
    test_data = []
    with open(test_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Loaded test set: {len(test_data)} samples")
    
    # Shuffle data for random distribution across GPUs
    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(test_data)
        print(f"Data shuffled with seed {args.seed}")
    else:
        print(f"Data not shuffled (sequential split)")
    
    model, processor = load_model(args.model_path, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = f"result_lvip_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output)
    
    # Run inference
    detailed_results = []

    if args.test_mode:
        test_data = test_data[:5]
        print("Running in test mode: only evaluating on 5 samples.")

    for idx, entry in enumerate(tqdm(test_data, desc="Evaluation")):
        try:
            # Construct input
            messages = make_chat_input(entry, image_root)
            
            # Call model
            response = call_model(model, processor, messages)
            
            # Extract answer
            num_options = len(entry.get('options', []))
            
            # Use LLM extraction if enabled, otherwise use rule-based
            if args.use_llm_extraction and openai_client:
                extracted_answer, extraction_success = extract_answer_with_llm(response, openai_client, num_options)
            else:
                extracted_answer, extraction_success = extract_answer(response, num_options)
            
            # Verify correctness
            ground_truth = entry['solution']
            is_correct = verify_answer(extracted_answer, ground_truth)
            
            # Record result
            detailed_results.append({
                "problem_id": entry['problem_id'],
                "data_source": entry.get('data_source', 'Unknown'),
                "reasoning_category": entry['reasoning_category'],
                "ground_truth": ground_truth,
                "model_response": response,
                "extracted_answer": extracted_answer,
                "extraction_success": extraction_success,
                "is_correct": is_correct
            })
            
        except Exception as e:
            print(f"\nWarning: Sample {entry['problem_id']} failed: {e}")
            # Record failed result
            detailed_results.append({
                "problem_id": entry['problem_id'],
                "data_source": entry.get('data_source', 'Unknown'),
                "reasoning_category": entry['reasoning_category'],
                "ground_truth": entry['solution'],
                "model_response": f"ERROR: {str(e)}",
                "extracted_answer": None,
                "extraction_success": False,
                "is_correct": False
            })
    
        # Final save to ensure all results are written
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    results = {
        "Fluid Intelligence": [x for x in detailed_results if x['reasoning_category'] == 'Fluid Intelligence'],
        "Crystallized Intelligence": [x for x in detailed_results if x['reasoning_category'] == 'Crystallized Intelligence'],
        "Visuospatial Cognition": [x for x in detailed_results if x['reasoning_category'] == 'Visuospatial Cognition'],
        "Mental Simulation": [x for x in detailed_results if x['reasoning_category'] == 'Mental Simulation'],
        "Visual Routines": [x for x in detailed_results if x['reasoning_category'] == 'Visual Routines'] 
    }

    overall_accuracy = sum(1 for x in detailed_results if x['is_correct']) / len(detailed_results)

    accuracy = {
        "Fluid Intelligence": sum(1 for x in results["Fluid Intelligence"] if x['is_correct']) / len(results["Fluid Intelligence"]) if results["Fluid Intelligence"] else 0,
        "Crystallized Intelligence": sum(1 for x in results["Crystallized Intelligence"] if x['is_correct']) / len(results["Crystallized Intelligence"]) if results["Crystallized Intelligence"] else 0,
        "Visuospatial Cognition": sum(1 for x in results["Visuospatial Cognition"] if x['is_correct']) / len(results["Visuospatial Cognition"]) if results["Visuospatial Cognition"] else 0,
        "Mental Simulation": sum(1 for x in results["Mental Simulation"] if x['is_correct']) / len(results["Mental Simulation"]) if results["Mental Simulation"] else 0,
        "Visual Routines": sum(1 for x in results["Visual Routines"] if x['is_correct']) / len(results["Visual Routines"]) if results["Visual Routines"] else 0
    }

    extraction_rate = sum(1 for x in detailed_results if x['extraction_success']) / len(detailed_results)

    extraction_rates = {
        "Fluid Intelligence": sum(1 for x in results["Fluid Intelligence"] if x['extraction_success']) / len(results["Fluid Intelligence"]) if results["Fluid Intelligence"] else 0,
        "Crystallized Intelligence": sum(1 for x in results["Crystallized Intelligence"] if x['extraction_success']) / len(results["Crystallized Intelligence"]) if results["Crystallized Intelligence"] else 0,
        "Visuospatial Cognition": sum(1 for x in results["Visuospatial Cognition"] if x['extraction_success']) / len(results["Visuospatial Cognition"]) if results["Visuospatial Cognition"] else 0,
        "Mental Simulation": sum(1 for x in results["Mental Simulation"] if x['extraction_success']) / len(results["Mental Simulation"]) if results["Mental Simulation"] else 0,
        "Visual Routines": sum(1 for x in results["Visual Routines"] if x['extraction_success']) / len(results["Visual Routines"]) if results["Visual Routines"] else 0
    }   

    output_md_path = os.path.join(args.output_dir, f"report_lvip_{timestamp}.md")

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# CogSense-Bench Evaluation Report: {args.model_path}\n\n")
        
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Total Samples**: {len(detailed_results)}\n")
        f.write(f"- **Overall Accuracy**: {overall_accuracy:.1%} ({sum(1 for x in detailed_results if x['is_correct'])}/{len(detailed_results)})\n")
        f.write(f"- **Answer Extraction Success Rate**: {extraction_rate:.1%} ({sum(1 for x in detailed_results if x['extraction_success'])}/{len(detailed_results)})\n\n")

        f.write("## Accuracy by reasoning_category\n\n")
        f.write("|    Category    | Samples | Correct | Accuracy | Extraction Rate |\n")
        f.write("|----------------|---------|---------|----------|-----------------|\n")
        
        for category in ["Fluid Intelligence", "Crystallized Intelligence", "Visuospatial Cognition", "Mental Simulation", "Visual Routines"]:
            total = len(results[category])
            correct = sum(1 for x in results[category] if x['is_correct'])
            acc = accuracy[category]
            ext_rate = extraction_rates[category]
            f.write(f"| {category:<12} | {total:7d} | {correct:7d} | {acc:8.1%} | {ext_rate:15.1%} |\n")
        
        f.write(f"\n## Detailed Results\n\n")
        f.write(f"Full results saved to: '{output_path}'\n")
    
if __name__ == "__main__":
    main()

