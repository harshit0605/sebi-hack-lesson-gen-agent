import os
import base64
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import socket
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Ensure your GOOGLE_API_KEY is set as an environment variable
PDF_FILE_PATH = (
    r"D:\Projects\gemma-impact\books\English\Class 3\English\cesa1dd\cesa103.pdf"
)
# MODEL_NAME = "gemini-3n-e4b-it"
# MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "gpt-4.1-mini"

# --- Image Optimization Settings ---
DPI = 96
JPEG_QUALITY = 75
MAX_DIMENSION = 1024


def optimize_image(image_bytes):
    """Resizes and converts an image to a smaller JPEG format."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


# --- Main Script ---
def analyze_full_story(file_path):
    """
    Processes an entire PDF, combines all text and images, and sends them
    in a single request for a holistic story analysis.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    print(f"Loading and processing all pages from PDF: {file_path}...")
    pdf_document = fitz.open(file_path)

    full_story_text = ""
    page_images_data = []
    total_original_size = 0
    total_optimized_size = 0

    # 1. Loop through the PDF to collect and optimize content from all pages
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        page_number = page_num + 1

        # Accumulate text with page separators
        full_story_text += f"\n\n--- Page {page_number} Text ---\n"
        full_story_text += page.get_text()

        # Render, optimize, and collect image data
        pix = page.get_pixmap(dpi=DPI)
        img_bytes = pix.tobytes("png")
        # Image.open(io.BytesIO(img_bytes)).show()
        optimized_bytes = optimize_image(img_bytes)
        # Image.open(io.BytesIO(optimized_bytes)).show()

        total_original_size += len(img_bytes)
        total_optimized_size += len(optimized_bytes)

        page_images_data.append(optimized_bytes)

    print(f"Finished processing {len(pdf_document)} pages.")
    print(f"Total original image payload: {total_original_size / 1024:.2f} KB")
    print(f"Total optimized image payload: {total_optimized_size / 1024:.2f} KB")

    try:
        # 2. Construct the single, large multimodal message
        # The message content starts with text, then is followed by all the images
        prompt_content = [
            {
                "type": "text",
                "text": "You are a story analyst. The following text and sequence of images represent a children's story from a PDF, with each image corresponding to a page. Read the entire story, look at all the pictures, and then provide a comprehensive summary. Explain the moral of the story.",
            },
            {"type": "text", "text": full_story_text},
        ]

        # Add all the collected images to the prompt content
        for img_data in page_images_data:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(img_data).decode('utf-8')}",
                    },
                }
            )

        message = HumanMessage(content=prompt_content)

        # 3. Initialize the model and make one API call
        print("\nInitializing model and sending the complete story for analysis...")
        llm = ChatOpenAI(model=MODEL_NAME)
        response = llm.with_config(configurable={"timeout": 300}).invoke(
            [message]
        )  # Increased timeout for larger payload

        # 4. Print the final analysis
        print("\n--- Comprehensive Story Analysis ---")
        print(response.content)

    except socket.timeout:
        print(
            "\n--- ERROR: The request timed out. The combined story payload may be too large even with optimization. ---"
        )
    except Exception as e:
        print(f"\n--- An error occurred: {e} ---")
    finally:
        pdf_document.close()


if __name__ == "__main__":
    analyze_full_story(PDF_FILE_PATH)
