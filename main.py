from flask import Flask, request, render_template, jsonify
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from google.generativeai import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv
import secrets
import re
from pdf2image import convert_from_path
import pytesseract

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)
model = GenerativeModel('gemini-1.5-flash')

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    """Sanitize filename to remove problematic characters and ensure uniqueness."""
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    random_suffix = secrets.token_hex(4)
    name, ext = os.path.splitext(safe_filename)
    return f"{name}_{random_suffix}{ext}"

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using PyPDF2, with OCR fallback for image-based PDFs."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if pdf_reader.is_encrypted:
                logger.error("PDF is encrypted and cannot be processed")
                return "Error: PDF is encrypted and cannot be processed without a password"
            text = ''
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ''
                text += page_text
                logger.debug(f"PyPDF2 - Page {page_num}: Extracted {len(page_text)} characters")
            if text.strip():
                logger.info(f"Successfully extracted {len(text)} characters using PyPDF2")
                return text
            else:
                logger.warning("No text extracted with PyPDF2. Falling back to OCR.")
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PdfReadError during PyPDF2 extraction: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during PyPDF2 extraction: {str(e)}")

    try:
        images = convert_from_path(file_path)
        logger.info(f"Converted PDF to {len(images)} images for OCR")
        text = ''
        for page_num, image in enumerate(images, 1):
            page_text = pytesseract.image_to_string(image)
            text += page_text
            logger.debug(f"OCR - Page {page_num}: Extracted {len(page_text)} characters")
        if not text.strip():
            logger.error("No text extracted from PDF using OCR.")
            return "Error: No text could be extracted from the PDF."
        logger.info(f"Successfully extracted {len(text)} characters using OCR")
        return text
    except Exception as e:
        logger.error(f"Error during OCR extraction: {str(e)}")
        return f"Error: Failed to extract text using OCR: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    try:
        text = docx2txt.process(file_path)
        logger.info(f"Successfully extracted {len(text)} characters from DOCX")
        return text
    except Exception as e:
        logger.error(f"Error extracting DOCX: {str(e)}")
        return f"Error: Failed to extract text from DOCX: {str(e)}"

def extract_text_from_txt(file_path):
    """Extract text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"Successfully extracted {len(text)} characters from TXT")
        return text
    except Exception as e:
        logger.error(f"Error extracting TXT: {str(e)}")
        return f"Error: Failed to extract text from TXT: {str(e)}"

def extract_text(file_path):
    """Extract text based on file extension."""
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        logger.error(f"Unsupported file type: {file_path}")
        return "Error: Unsupported file type"

def cleanup_files(file_paths):
    """Remove uploaded files after analysis."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
            else:
                logger.warning(f"File not found for deletion: {file_path}")
        except OSError as e:
            logger.warning(f"Error deleting file {file_path}: {str(e)}")

def analyze_content_with_gemini(resume_content, job_description):
    """Analyze resume content with Gemini API."""
    try:
        if not resume_content or not resume_content.strip():
            logger.error("Resume content is empty or invalid")
            return 0, "Error: Resume content is empty or invalid", []
        
        prompt = f"""
        You are a strict and practical career advisor evaluating a resume based on a provided job description. Perform the following tasks:
        1. Assign a score from 0 to 100 based on how well the resume matches the job description, considering relevance of skills, experience, clarity, and coherence.
        2. Provide a brief explanation for the score, highlighting strengths and weaknesses.
        3. Suggest 2-3 specific tips for improving the resume to better align with the job description.

        Job Description: {job_description}
        Resume Content: {resume_content}

        Return the response in the following format:
        Score: [number]
        Explanation: [text]
        Improvement Tips:
        - [Tip 1]
        - [Tip 2]
        - [Tip 3]
        """
        response = model.generate_content(prompt)
        response_text = response.text

        score = int(response_text.split('Score: ')[1].split('\n')[0])
        explanation = response_text.split('Explanation: ')[1].split('Improvement Tips:')[0].strip()
        tips = response_text.split('Improvement Tips:')[1].strip().split('\n- ')[1:]
        tips = [tip.strip() for tip in tips if tip.strip()]

        logger.info(f"Gemini analysis completed. Score: {score}")
        return score, explanation, tips
    except Exception as e:
        logger.error(f"Error analyzing content with Gemini: {str(e)}")
        return 0, f"Error analyzing content with Gemini: {str(e)}", []

@app.route('/')
def home():
    """Render the home page for multiple resume matching."""
    return render_template('home.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    """Match multiple resumes against a job description using TF-IDF."""
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')
        uploaded_files = []

        try:
            if not resume_files or not job_description:
                logger.warning("Missing resumes or job description")
                return render_template('home.html', message="Please upload resumes and enter a job description.")

            resumes = []
            valid_resume_files = []
            for resume_file in resume_files:
                if not allowed_file(resume_file.filename):
                    logger.warning(f"Invalid file type: {resume_file.filename}")
                    continue
                filename = sanitize_filename(resume_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                resume_file.save(file_path)
                uploaded_files.append(file_path)
                text = extract_text(file_path)
                if not text.startswith('Error'):
                    resumes.append(text)
                    valid_resume_files.append(resume_file)
                else:
                    logger.error(f"Failed to extract text from {filename}: {text}")

            if not resumes:
                return render_template('home.html', message="No valid resumes could be processed.")

            vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
            vectors = vectorizer.toarray()
            job_vector = vectors[0]
            resume_vectors = vectors[1:]
            similarities = cosine_similarity([job_vector], resume_vectors)[0]

            top_indices = similarities.argsort()[-5:][::-1]
            top_resumes = [valid_resume_files[i].filename for i in top_indices]
            similarity_scores = [round(similarities[i], 2) for i in top_indices]
            # Combine resumes and scores into a list of tuples
            resume_score_pairs = list(zip(top_resumes, similarity_scores))

            return render_template('home.html', message="Top matching resumes:", resume_score_pairs=resume_score_pairs)

        finally:
            cleanup_files(uploaded_files)

    return render_template('home.html')

@app.route('/analysis')
def analysis():
    """Render the analysis page for single resume evaluation."""
    return render_template('analysis.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Analyze a single resume using Gemini API."""
    if 'file' not in request.files or 'job_description' not in request.form:
        logger.warning("Missing file or job description in request")
        return jsonify({'error': 'Missing file or job description'}), 400

    file = request.files['file']
    job_description = request.form['job_description']
    uploaded_files = []

    if file.filename == '':
        logger.warning("No file selected")
        return jsonify({'error': 'No selected file'}), 400
    if not job_description:
        logger.warning("No job description provided")
        return jsonify({'error': 'No job description provided'}), 400
    if file and allowed_file(file.filename):
        filename = sanitize_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            uploaded_files.append(file_path)
            logger.info(f"File saved: {file_path}")

            content = extract_text(file_path)
            if content.startswith('Error'):
                logger.error(content)
                return jsonify({'error': content}), 400
            
            score, explanation, tips = analyze_content_with_gemini(content, job_description)
            if score == 0 and explanation.startswith('Error'):
                return jsonify({'error': explanation}), 400
            
            return jsonify({
                'content': content[:1000],
                'score': score,
                'explanation': explanation,
                'improvement_tips': tips
            })
        finally:
            cleanup_files(uploaded_files)
    logger.warning("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)