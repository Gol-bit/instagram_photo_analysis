# OpenAI-driven Instagram Image Analyser on AWS Pipeline

A Python-based AI agent running on an AWS instance that:
- **Reads** Instagram photos from S3 (`ig_pics/`).
- **Batches** them (200 images per batch) and sends to OpenAI Vision API (`gpt-4o-mini`).
- **Parses** the JSON responses into structured CSV.
- **Stores** results back in S3 (`results/image_analysis_results.csv`).
- **Notifies** progress and errors via Telegram bot.

## Analyzed Categories

The script classifies each Instagram photo into the following categories:
- **Primary category**:  
  `socializing`, `romantic picture`, `unique location`, `supporting cause`,  
  `face shot`, `special occasion`, `alone posing`, `playing sports`,  
  `family`, `interests`, `humorous shot`
- **Alternate category** (`category_alt`):  
  `travel`, `selfie`, `portrait`, `sports`, `pets`, `fashion`, `food`,  
  `artistic`, `funny`, `meme`, `cause_support`, `hobby`, `work_related`,  
  `product_promo`, `intimate`, `mirror_photo`, `group_photo`,  
  `minimalist_aesthetic`, `luxury_lifestyle`

Additional visual features include mood, lighting, blur level, depth of field, color palette,  
indoor/outdoor, expression authenticity, status symbols, lifestyle, background, and composition cues.

## Purpose

This agent was built to support a scientific study of the relationship between  
users’ psychological traits and their Instagram content usage patterns.

## Requirements

- Python 3.8+  
- `boto3`, `openai`, `pandas`, `requests`, `loguru`  

## Configuration

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export OPENAI_API_KEY=...
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...

python instagram_pics_analyser_openai.py
