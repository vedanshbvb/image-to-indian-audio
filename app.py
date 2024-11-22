from flask import Flask, request, render_template, jsonify, send_file
import os
import subprocess
import requests
from gtts import gTTS
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import sys
from werkzeug.utils import secure_filename

sys.path.append("IndicTransToolkit/IndicTransToolkit")
from processor import IndicProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload and audio output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

# Global variables for model and processor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None
ip = None

def initialize_models():
    global tokenizer, model, ip
    
    ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
    quantization = None
    
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        if DEVICE == "cuda":
            model.half()

    model.eval()
    ip = IndicProcessor(inference=True)

def batch_translate(input_sentences, src_lang, tgt_lang, device="cpu"):
    translations = []
    for i in range(0, len(input_sentences), 4):
        batch = input_sentences[i : i + 4]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        torch.cuda.empty_cache()

    return translations

def generate_audio(translations, lang, output_prefix):
    audio_files = []
    for i, text in enumerate(translations):
        output_path = f"static/audio/{output_prefix}_{i+1}.mp3"
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        audio_files.append(output_path)
    return audio_files

def get_image_caption(image_path):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    headers = {"Authorization": "Bearer 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"}

    with open(image_path, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()[0]['generated_text']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save uploaded image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Get image caption
        caption = get_image_caption(filepath)

        # Translate caption
        languages = {"hin_Deva": "hi", "pan_Guru": "pa", "guj_Gujr": "gu"}
        translations = {}
        audio_files = {}

        for lang_code, lang in languages.items():
            translations[lang] = batch_translate([caption], "eng_Latn", lang_code, DEVICE)
            audio_files[lang] = generate_audio(translations[lang], lang, f"audio_{lang}")

        return jsonify({
            'caption': caption,
            'translations': translations,
            'audio_files': audio_files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded image
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True)