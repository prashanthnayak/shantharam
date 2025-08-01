import requests
import numpy as np
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import cv2
import os
from datetime import datetime
import sys
import torchvision
# import whisper
# import torch

# Global variables for model caching
_processor = None
_model = None

def get_cached_model():
    """Cache the OCR model to avoid reloading it every time"""
    global _processor, _model
    if _processor is None or _model is None:
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        import torch
        from PIL import Image
        
        print("Loading OCR model (first time only)...")
        model_name = 'anuashok/ocr-captcha-v3'
        _processor = TrOCRProcessor.from_pretrained(model_name)
        _model = VisionEncoderDecoderModel.from_pretrained(model_name)
        print("OCR model loaded and cached!")
    
    return _processor, _model

def create_chrome_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    
    # Performance optimizations
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")  # Don't load images
    chrome_options.add_argument("--disable-javascript")  # Disable JS if not needed
    chrome_options.add_argument("--disable-css")  # Disable CSS if not needed
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    chrome_options.add_argument("--disable-field-trial-config")
    chrome_options.add_argument("--disable-ipc-flooding-protection")
    
    if sys.platform == "darwin":
        chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif sys.platform.startswith("linux"):
        chrome_options.binary_location = "/usr/bin/chromium-browser"
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("Chrome driver created successfully.")
        return driver
    except Exception as e:
        print(f"Error creating driver: {e}")
        return None

# Function to save CAPTCHA image (optimized)
def save_captcha_image(driver, save_directory=os.path.expanduser("~/Desktop/captcha_images")):
    wait = WebDriverWait(driver, 10)  # Reduced from 30 to 10 seconds
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"Created directory: {save_directory}")
        
        captcha_img = wait.until(EC.presence_of_element_located((By.ID, 'captcha_image')))
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captcha_{timestamp}.png"
        filepath = os.path.join(save_directory, filename)

        # Take a screenshot of the CAPTCHA element
        captcha_img.screenshot(filepath)
        print(f"CAPTCHA image saved as: {filepath}")
        
        return filepath

    except PermissionError as e:
        print(f"Permission denied: {e}")
        return None
    except Exception as e:
        print(f"Failed to save CAPTCHA image: {e}")
        return None

# Optimized OCR function
def solve_captcha_ocr(image_path):
    """Optimized OCR function with cached model"""
    try:
        processor, model = get_cached_model()
        
        from PIL import Image
        image = Image.open(image_path).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        combined = Image.alpha_composite(background, image).convert("RGB")

        pixel_values = processor(combined, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return ocr_text
        
    except Exception as e:
        print(f"OCR failed: {e}")
        return None

# Main script (optimized)
def main():
    driver = create_chrome_driver(headless=True)  # Use headless for speed
    if driver:
        try:
            print("Loading website...")
            driver.get("https://services.ecourts.gov.in/ecourtindia_v6/")
            time.sleep(2)  # Reduced from 5 to 2 seconds

            # Enter CNR
            wait = WebDriverWait(driver, 10)  # Reduced from 15 to 10 seconds
            cnr_input = wait.until(EC.presence_of_element_located((By.ID, 'cino')))
            cnr_number = "KAUP050003552024"
            cnr_input.clear()
            cnr_input.send_keys(cnr_number)
            print(f"Entered CNR number: {cnr_number}")

            # Save CAPTCHA image
            saved_image_path = save_captcha_image(driver)

            if saved_image_path:
                print(f"CAPTCHA image saved to: {saved_image_path}")
                print("Solving CAPTCHA...")

                # Solve CAPTCHA with cached model
                ocr_text = solve_captcha_ocr(saved_image_path)
                
                if ocr_text and len(ocr_text.strip()) > 0:
                    generated_text = ocr_text
                    print(f"CAPTCHA solved: {generated_text}")
                else:
                    print("OCR failed. Cannot proceed.")
                    return

                # Submit CAPTCHA
                try:
                    captcha_input = driver.find_element(By.ID, 'fcaptcha_code')
                    captcha_input.clear()
                    captcha_input.send_keys(generated_text)
                    print(f"Entered CAPTCHA: {generated_text}")

                    submit_button = wait.until(EC.element_to_be_clickable((By.ID, 'searchbtn')))
                    submit_button.click()
                    print("Search submitted. Waiting for results...")
                    time.sleep(3)  # Reduced from 5 to 3 seconds
                    
                    # Extract case history
                    extract_case_history(driver)
                    
                except Exception as e:
                    print(f"Failed to submit CAPTCHA: {e}")

            else:
                print("Failed to save CAPTCHA image.")

        except Exception as e:
            print(f"Script error: {e}")
        finally:
            print("Closing browser...")
            driver.quit()
            print("Browser closed.")
    else:
        print("Failed to create Chrome driver.")

def extract_case_history(driver):
    """Extract case history data"""
    try:
        print("Extracting case history...")
        
        wait = WebDriverWait(driver, 10)  # Reduced from 15 to 10 seconds
        
        case_history = []
        try:
            history_table = driver.find_element(By.CLASS_NAME, 'history_table')
            rows = history_table.find_elements(By.TAG_NAME, 'tr')
            
            print(f"\n{'='*60}")
            print("CASE HISTORY TABLE")
            print(f"{'='*60}")
            print(f"{'Judge':<30} {'Business Date':<12} {'Hearing Date':<12} {'Purpose':<15}")
            print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*15}")
            
            for row in rows[1:]:  # Skip header row
                cells = row.find_elements(By.TAG_NAME, 'td')
                if len(cells) >= 4:
                    judge = cells[0].text.strip()
                    business_date = cells[1].text.strip()
                    hearing_date = cells[2].text.strip()
                    purpose = cells[3].text.strip()
                    
                    history_entry = {
                        'Judge': judge,
                        'Business_on_Date': business_date,
                        'Hearing_Date': hearing_date,
                        'Purpose_of_Hearing': purpose
                    }
                    case_history.append(history_entry)
                    
                    # Print formatted row
                    print(f"{judge:<30} {business_date:<12} {hearing_date:<12} {purpose:<15}")
            
            print(f"{'='*60}")
            print(f"Total History Entries: {len(case_history)}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Could not extract case history: {e}")
        
        # Save to CSV
        if case_history:
            save_to_csv(case_history)
        
    except Exception as e:
        print(f"Failed to extract case data: {e}")

def save_to_csv(case_history):
    """Save case history to CSV file"""
    try:
        import csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"case_history_KAUP050003552024_{timestamp}.csv"
        
        save_directory = "/Users/prashanth/Desktop/shantharam"
        full_path = os.path.join(save_directory, csv_filename)
        print(f"\nSaving CSV file to: {full_path}")
        
        with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Judge', 'Business_on_Date', 'Hearing_Date', 'Purpose_of_Hearing'])
            
            # Write data
            for entry in case_history:
                writer.writerow([
                    entry['Judge'],
                    entry['Business_on_Date'],
                    entry['Hearing_Date'],
                    entry['Purpose_of_Hearing']
                ])
        
        print(f"Case history saved to: {csv_filename}")
        
    except Exception as e:
        print(f"Failed to save CSV: {e}")

if __name__ == "__main__":
    main() 