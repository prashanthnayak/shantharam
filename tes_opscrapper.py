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
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image
import csv

# Preload OCR model once for efficiency
model_name = 'anuashok/ocr-captcha-v3'
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
print("OCR model preloaded.")

def create_chrome_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")
    # Disable images for faster loading
    chrome_options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    # Set eager page load strategy
    chrome_options.page_load_strategy = 'eager'  # Faster: wait only until interactive [7]
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

# Function to save CAPTCHA image (optimized with explicit wait)
def save_captcha_image(driver, save_directory=os.path.expanduser("~/Desktop/captcha_images")):
    wait = WebDriverWait(driver, 5)  # Reduced timeout [13]
    try:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            print(f"Created directory: {save_directory}")
        
        captcha_img = wait.until(EC.presence_of_element_located((By.ID, 'captcha_image')))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captcha_{timestamp}.png"
        filepath = os.path.join(save_directory, filename)
        
        captcha_img.screenshot(filepath)
        print(f"CAPTCHA image saved as: {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to save CAPTCHA image: {e}")
        return None

# Main script
driver = create_chrome_driver(headless=True)  # Use headless for speed [1][14]
if driver:
    try:
        driver.get("https://services.ecourts.gov.in/ecourtindia_v6/")
        
        # Enter CNR with explicit wait
        wait = WebDriverWait(driver, 5)  # Optimized wait [2][13]
        cnr_input = wait.until(EC.presence_of_element_located((By.ID, 'cino')))
        cnr_number = "KAUP050003552024"
        cnr_input.clear()
        cnr_input.send_keys(cnr_number)
        print(f"Entered CNR number: {cnr_number}")
        
        # Save CAPTCHA image
        saved_image_path = save_captcha_image(driver)
        
        if saved_image_path:
            print(f"CAPTCHA image successfully saved to: {saved_image_path}")
            print("Attempting to solve CAPTCHA...")
            
            # OCR solving (using preloaded model)
            try:
                image = Image.open(saved_image_path).convert("RGBA")
                background = Image.new("RGBA", image.size, (255, 255, 255))
                combined = Image.alpha_composite(background, image).convert("RGB")
                
                pixel_values = processor(combined, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print("Predicted CAPTCHA text (OCR):", ocr_text)
                
                if ocr_text and len(ocr_text.strip()) > 0:
                    generated_text = ocr_text
                    print(f"Using OCR result: {generated_text}")
                else:
                    print("OCR failed. Cannot proceed.")
                    generated_text = "FAILED"
            except Exception as e:
                print(f"OCR failed: {e}")
                generated_text = "FAILED"
            
            # Enter CAPTCHA and submit
            try:
                captcha_input = driver.find_element(By.ID, 'fcaptcha_code')
                captcha_input.clear()
                captcha_input.send_keys(generated_text)
                print(f"Entered solved CAPTCHA: {generated_text}")
                
                submit_button = wait.until(EC.element_to_be_clickable((By.ID, 'searchbtn')))
                submit_button.click()
                print("Search button clicked. Waiting for results...")
                
                # Extract case history with explicit wait (no fixed sleep)
                print("Extracting case history data...")
                history_table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'history_table')))
                rows = history_table.find_elements(By.TAG_NAME, 'tr')
                
                case_history = []
                print(f"\n{'='*80}")
                print("CASE HISTORY TABLE")
                print(f"{'='*80}")
                print(f"{'Judge':<40} {'Business Date':<15} {'Hearing Date':<15} {'Purpose':<20}")
                print(f"{'-'*40} {'-'*15} {'-'*15} {'-'*20}")
                
                for row in rows[1:]:  # Skip header
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
                        
                        print(f"{judge:<40} {business_date:<15} {hearing_date:<15} {purpose:<20}")
                
                print(f"{'='*80}")
                print(f"Total History Entries: {len(case_history)}")
                print(f"{'='*80}")
                
                # Save to CSV
                if case_history:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_filename = f"case_history_{cnr_number}_{timestamp}.csv"
                    save_directory = "/Users/prashanth/Desktop/shantharam"
                    full_path = os.path.join(save_directory, csv_filename)
                    print(f"\nSaving CSV file to: {full_path}")
                    
                    with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Judge', 'Business_on_Date', 'Hearing_Date', 'Purpose_of_Hearing'])
                        for entry in case_history:
                            writer.writerow([
                                entry['Judge'],
                                entry['Business_on_Date'],
                                entry['Hearing_Date'],
                                entry['Purpose_of_Hearing']
                            ])
                    print(f"Case history saved to: {csv_filename}")
            except Exception as e:
                print(f"Failed to submit CAPTCHA and search: {e}")
        else:
            print("Failed to save CAPTCHA image.")
    except Exception as e:
        print(f"Script error: {e}")
    finally:
        driver.quit()
        print("Browser session closed.")
else:
    print("Failed to create Chrome driver.")
