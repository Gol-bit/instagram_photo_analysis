import os
import json
import base64
import time
import requests
import pandas as pd
import boto3
from openai import OpenAI
from loguru import logger as log

# === AWS Settings ===
BUCKET_NAME = "my-igpop-photo-bucket-2025"
RESULTS_KEY = "results/image_analysis_results.csv"
LOCAL_FOLDER = "/tmp/s3_images"
os.makedirs(LOCAL_FOLDER, exist_ok=True)

s3 = boto3.client("s3")
aws_requests = {
    "list_objects_v2": 0,
    "download_file": 0,
    "upload_file": 0
}

# OpenAI API key from environment variable
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# === Telegram Notifications ===
def send_telegram_message(message):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set!")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
        log.info(f"[TG] {message}")
    except Exception as e:
        log.error(f"[TG] Exception: {e}")

# === CSV Handling ===
def load_existing_csv():
    local_csv = "current_results.csv"
    try:
        s3.download_file(BUCKET_NAME, RESULTS_KEY, local_csv)
        aws_requests["download_file"] += 1
        return pd.read_csv(local_csv)
    except Exception:
        return pd.DataFrame()

def save_csv_to_s3(df):
    local_csv = "current_results.csv"
    df.to_csv(local_csv, index=False)
    s3.upload_file(local_csv, BUCKET_NAME, RESULTS_KEY)
    aws_requests["upload_file"] += 1

# === List folders in S3 ===
def list_folders_in_s3(prefix):
    aws_requests["list_objects_v2"] += 1
    result = set()
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter="/")
    if "CommonPrefixes" in resp:
        for cp in resp["CommonPrefixes"]:
            result.add(cp["Prefix"])
    else:
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        for obj in resp.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) > 1:
                result.add(f"{parts[0]}/{parts[1]}/")
    return sorted(result)

# === Download Images from S3 ===
def download_images_from_s3(prefix):
    aws_requests["list_objects_v2"] += 1
    resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    local_folder = os.path.join(LOCAL_FOLDER, prefix.replace('/', '_'))
    os.makedirs(local_folder, exist_ok=True)
    images = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = os.path.basename(key)
            local_path = os.path.join(local_folder, filename)
            s3.download_file(BUCKET_NAME, key, local_path)
            aws_requests["download_file"] += 1
            images.append((prefix, local_path))
    return images

# === JSONL Entry Generation ===
def generate_jsonl_entry(prefix, image_path):
    try:
        with open(image_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        error_msg = f"[ERROR] Failed to open/encode {image_path}: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)
        return None

    prompt_text = """
Please analyze this image based on the following criteria and return exactly a valid JSON object without any markdown formatting, code fences, or additional text.

{
  "people_count": <number>,
  "faces_count": <number>,
  "person_details": [
      {
      "gender": "<male/female/unknown>",
      "makeup_level": <0-5>,
      "smiling": "<yes/no>",
      "stylish_clothes_level": <0-5>,
      "brand_logos": "<yes/no>",
      "expensive_clothes_level": <0-5>,
      "provocativeness_level": <-3 to 3>,
      "neatness_level": <0-5>,
      "physical_attractiveness_level": <-3 to 3>,
      "wearing_glasses": "<yes/no>",
      "dominant_emotion": "<anger/contempt/disgust/fear/happiness/neutral/sadness/surprise>",
      "posing_level": <0-5>,
      "tattoos": "<yes/no>",
      "piercing": "<yes/no>",
      "flashy_clothing": "<yes/no>",
      "appearance_style": "<urban/sporty/business/evening/casual/streetwear/grunge/hipster/punk/goth/military/retro/minimalist/classic/luxury>",
      "nudity": <0-5>,
      "pose": "<dominant/confident/aggressive/relaxed/posing_formal/flirtatious/vulnerable/defensive/natural/playful/assertive>"
      }
  ],
  "is_selfie": "<yes/no>",
  "is_ads": "<yes/no>",
  "filter_or_editing_level": <0-5>,
  "description": "<string>",
  "category": "<one of: socializing, romantic picture, unique location, supporting cause, face shot, special occasion, alone posing, playing sports, family, interests, humorous shot>",
  "category_alt": "<one of: socializing, romantic, travel, special_event, selfie, portrait, sports, family, pets, fashion, food, artistic, funny, meme, cause_support, hobby, work_related, product_promo, intimate, mirror_photo, group_photo, minimalist_aesthetic, luxury_lifestyle>",
  "photo_mood": "<cheerful/serious/dramatic/romantic/anxious/mysterious/aggressive/dynamic/calm>",
  "lighting": "<natural/artificial/mixed/contrasty/diffused/low-key/high-key>",
  "blur_level": <0-5>,
  "depth_of_field": "<shallow/deep>",
  "color_palette": "<monochrome/warm/cold/pastel/acidic/contrasting/neutral>",
  "interior_vs_exterior": "<indoor/outdoor/hard to tell>",
  "expression_authenticity": <0-5>,
  "status_symbols_present": "<yes/no>",
  "lifestyle": "<one of: productivity, sports, travel, fitness, party, career, education, culture, romantic, pet_life, luxury_lifestyle, family_oriented, creative_artistic, activist_or_cause, influencer_style, daily_life, adventurous, spiritual, laziness>",
  "background": "<one of: home_private, home_decorated, messy_home, nature, urban_city, party_scene, cultural_place, gym_or_sports_area, workspace, educational_place, luxury_location, vehicle_or_transport, mirror_selfie, minimalist, cluttered_or_noisy, unclear_or_generic, studio_or_staged>",
  "facial_expressions_intensity": <0-5>,
  "interaction_type": "<hugging/handshake/eye contact/talking/no interaction/dancing/sharing an object/laughing together/running together/posing together/other>",
  "cultural_elements": "<yes/no>",
  "composition": {
      "balanced": "<yes/no>",
      "leading_lines": "<yes/no>",
      "framing": "<yes/no>",
      "negative_space": "<yes/no>",
      "focal_point": "<string>"
  }
}
Return the result as strict JSON only. Do not include any commentary, explanation, or markdown formatting. All keys and values must be enclosed in double quotes. Use commas to separate all elements. Do not omit any quotes. Do not include trailing commas.
"""
    entry = {
        "custom_id": f"{prefix.strip('/')}/{os.path.basename(image_path)}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                    ]
                }
            ],
            "max_tokens": 800
        }
    }
    return entry

# === Parse Results and Write to CSV ===
def parse_response_content(content, custom_id):
    result = {
        "folder_name": custom_id.split("/")[0],
        "image_name": custom_id.split("/")[-1],
        "people_count": content.get("people_count", ""),
        "faces_count": content.get("faces_count", ""),
        "is_selfie": content.get("is_selfie", ""),
        "is_ads": content.get("is_ads", ""),
        "filter_or_editing_level": content.get("filter_or_editing_level", ""),
        "description": content.get("description", ""),
        "category": content.get("category", ""),
        "category_alt": content.get("category_alt", ""),
        "photo_mood": content.get("photo_mood", ""),
        "lighting": content.get("lighting", ""),
        "blur_level": content.get("blur_level", ""),
        "depth_of_field": content.get("depth_of_field", ""),
        "color_palette": content.get("color_palette", ""),
        "facial_expressions_intensity": content.get("facial_expressions_intensity", ""),
        "interaction_type": content.get("interaction_type", ""),
        "cultural_elements": content.get("cultural_elements", ""),
        "interior_vs_exterior": content.get("interior_vs_exterior", ""),
        "expression_authenticity": content.get("expression_authenticity", ""),
        "status_symbols_present": content.get("status_symbols_present", ""),
        "lifestyle": content.get("lifestyle", ""),
        "background": content.get("background", ""),
        "composition_balanced": content.get("composition", {}).get("balanced", ""),
        "composition_leading_lines": content.get("composition", {}).get("leading_lines", ""),
        "composition_framing": content.get("composition", {}).get("framing", ""),
        "composition_negative_space": content.get("composition", {}).get("negative_space", ""),
        "composition_focal_point": content.get("composition", {}).get("focal_point", "")
    }
    # up to 5 people
    person_details = content.get("person_details", [])
    for i in range(5):
        pd_item = person_details[i] if i < len(person_details) else {}
        if not isinstance(pd_item, dict):
            try:
                pd_item = json.loads(pd_item)
            except:
                pd_item = {}
        result.update({
            f"Person{i+1}_Gender": pd_item.get("gender", ""),
            f"Person{i+1}_MakeupLevel": pd_item.get("makeup_level", ""),
            f"Person{i+1}_Smiling": pd_item.get("smiling", ""),
            f"Person{i+1}_StylishClothesLevel": pd_item.get("stylish_clothes_level", ""),
            f"Person{i+1}_BrandLogos": pd_item.get("brand_logos", ""),
            f"Person{i+1}_ExpensiveClothesLevel": pd_item.get("expensive_clothes_level", ""),
            f"Person{i+1}_ProvocativenessLevel": pd_item.get("provocativeness_level", ""),
            f"Person{i+1}_NeatnessLevel": pd_item.get("neatness_level", ""),
            f"Person{i+1}_AttractivenessLevel": pd_item.get("physical_attractiveness_level", ""),
            f"Person{i+1}_WearingGlasses": pd_item.get("wearing_glasses", ""),
            f"Person{i+1}_DominantEmotion": pd_item.get("dominant_emotion", ""),
            f"Person{i+1}_PosingLevel": pd_item.get("posing_level", ""),
            f"Person{i+1}_Tattoos": pd_item.get("tattoos", ""),
            f"Person{i+1}_Piercing": pd_item.get("piercing", ""),
            f"Person{i+1}_FlashyClothing": pd_item.get("flashy_clothing", ""),
            f"Person{i+1}_AppearanceStyle": pd_item.get("appearance_style", ""),
            f"Person{i+1}_Nudity": pd_item.get("nudity", ""),
            f"Person{i+1}_Pose": pd_item.get("pose", "")
        })
    return result

# === Main Logic ===
def main():
    send_telegram_message("🚀 Starting batch image analysis")
    df_all = load_existing_csv()
    processed = set(df_all["image_name"].astype(str)) if "image_name" in df_all.columns else set()

    all_images = []
    total_available = 0
    for folder in list_folders_in_s3("ig_pics/"):
        imgs = download_images_from_s3(folder)
        total_available += len(imgs)
        imgs = [(p, i) for (p, i) in imgs if os.path.basename(i) not in processed]
        all_images.extend(imgs)

    skipped = total_available - len(all_images)
    send_telegram_message(f"🔍 Skipped {skipped} already processed images, {len(all_images)} remain.")

    if not all_images:
        send_telegram_message("❌ No new images to analyze.")
        return

    BATCH_SIZE = 200
    chunks = [all_images[i:i+BATCH_SIZE] for i in range(0, len(all_images), BATCH_SIZE)]
    send_telegram_message(f"📸 {len(all_images)} images in {len(chunks)} batch(es).")

    processed_count = 0
    for idx, chunk in enumerate(chunks, start=1):
        send_telegram_message(f"📤 Processing batch {idx}/{len(chunks)}...")
        payload_path = f"batch_payload_{idx}.jsonl"
        with open(payload_path, "w") as f:
            for prefix, image_path in chunk:
                e = generate_jsonl_entry(prefix, image_path)
                if e:
                    f.write(json.dumps(e) + "\n")

        send_telegram_message(f"📤 Sending batch {idx} to OpenAI...")
        with open(payload_path, "rb") as f_in:
            input_file = client.files.create(file=f_in, purpose="batch")
        batch = client.batches.create(input_file_id=input_file.id,
                                      endpoint="/v1/chat/completions",
                                      completion_window="24h")
        batch_id = batch.id
        send_telegram_message(f"✅ Batch {idx} created (ID: {batch_id}), waiting for completion...")
        while True:
            status = client.batches.retrieve(batch_id)
            if status.status == "completed":
                break
            if status.status in {"failed", "expired"}:
                send_telegram_message(f"❌ Batch {idx} ended with status {status.status}")
                break
            time.sleep(60)

        output = client.files.retrieve_content(status.output_file_id)
        result_file = f"batch_results_{idx}.jsonl"
        with open(result_file, "w") as outf:
            outf.write(output)

        send_telegram_message(f"📥 Parsing results of batch {idx}...")
        rows = []
        for line in open(result_file):
            data = json.loads(line)
            if data.get("status") not in {None, "completed"}:
                continue
            raw = data.get("response", {}).get("body", {})\
                     .get("choices", [{}])[0]\
                     .get("message", {}).get("content", "{}")
            try:
                content = json.loads(raw)
            except Exception as e:
                send_telegram_message(f"[ERROR] JSON decode failed for {data.get('custom_id')}: {e}")
                continue
            rows.append(parse_response_content(content, data.get("custom_id", "")))

        if rows:
            df_new = pd.DataFrame(rows)
            df_all = pd.concat([df_all, df_new], ignore_index=True)
            save_csv_to_s3(df_all)
            processed_count += len(rows)
            pct = processed_count / len(all_images) * 100
            send_telegram_message(f"✅ Batch {idx} done: {processed_count}/{len(all_images)} ({pct:.2f}%)")
        else:
            send_telegram_message(f"⚠️ Batch {idx} yielded no results.")

    send_telegram_message(f"🚀 Analysis complete: {processed_count} images processed. AWS calls: {aws_requests}")

if __name__ == "__main__":
    main()
