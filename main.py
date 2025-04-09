from fastapi import FastAPI, BackgroundTasks, HTTPException, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl, Field
from bs4 import BeautifulSoup
import google.generativeai as genai
from typing import List, Dict, Optional, Any
from datetime import datetime
from urllib.parse import urlparse
import httpx
import re
import os
import csv
import json
import uuid
import asyncio
import pandas as pd
import aiofiles
import logging
import io

from dotenv import load_dotenv
load_dotenv()

# Initialize
app = FastAPI(title="Advanced Lead Generation Tool")

# Setup directories
EXPORT_DIR = "exports"
TEMPLATES_DIR = "templates"
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lead_gen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lead_gen")

# Configure Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY not set in environment. Create a .env file with GEMINI_API_KEY=your_key")
genai.configure(api_key=gemini_api_key)

MAX_CONCURRENT_REQUESTS = 5
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Models
class ScrapeRequest(BaseModel):
    urls: List[HttpUrl]
    extract_phones: bool = True
    extract_social_media: bool = True
    extract_company_info: bool = True
    deep_analysis: bool = False

class Contact(BaseModel):
    emails: List[str] = Field(default_factory=list)
    phones: List[str] = Field(default_factory=list)
    social_media: Dict[str, str] = Field(default_factory=dict)
    
class OrganizationInfo(BaseModel):
    name: Optional[str] = None
    industry: Optional[str] = None
    size_estimate: Optional[str] = None
    description: Optional[str] = None

class Lead(BaseModel):
    id: str
    url: str
    domain: str
    title: Optional[str] = None
    contact_info: Contact
    org_info: OrganizationInfo
    relevant: bool
    relevance_score: float = 0.0
    analysis: Optional[str] = None
    timestamp: str

class BatchUploadRequest(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class JobStatus(BaseModel):
    job_id: str
    status: str
    total_urls: int
    processed_urls: int
    completed: bool
    results_file: Optional[str] = None

active_jobs = {}

async def extract_text_from_url(url: str) -> Dict[str, Any]:
    try:
        async with request_semaphore:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                soup = BeautifulSoup(response.text, "html.parser")
                
                title = soup.title.string if soup.title else None
                
                for elem in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'li']):
                    if elem.string:
                        elem.string.replace_with(" " + elem.string + " ")
                
                text = soup.get_text(" ", strip=True)
                text = re.sub(r'\s+', ' ', text).strip()

                return {
                    "title": title,
                    "text": text,
                    "html": response.text
                }
    except Exception as e:
        logger.error(f"[SCRAPE ERROR] {url}: {e}")
        return {"title": None, "text": "", "html": ""}

def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text using a comprehensive regex pattern.
    This enhanced version catches more valid email formats.
    """
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    emails = re.findall(pattern, text)
    
    valid_emails = []
    for email in emails:
        if '.' in email.split('@')[1] and not email.endswith('.'):
            valid_emails.append(email)
    
    return list(set(valid_emails))

def extract_phones(text: str) -> List[str]:
    patterns = [
        r"\+\d{1,3}\s?[\(\-\s]?\d{1,4}[\)\-\s]?\s?\d{1,4}[\-\s]?\d{1,4}",  # +1 (123) 456-7890
        r"\(\d{3}\)\s?\d{3}[-\s]?\d{4}",  # (123) 456-7890
        r"\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}",  # 123-456-7890
        r"\d{5}\s\d{5,6}",  # 12345 123456
        r"\d{4}\s\d{3}\s\d{3}"  # 1234 567 890
    ]
    
    phone_numbers = []
    for pattern in patterns:
        phone_numbers.extend(re.findall(pattern, text))
    
    cleaned_numbers = []
    for phone in phone_numbers:
        digits = re.sub(r'[^0-9+]', '', phone)
        if len(digits) >= 7: 
            cleaned_numbers.append(phone)
    
    return list(set(cleaned_numbers))

def extract_social_media(html: str) -> Dict[str, str]:
    social_patterns = {
        "linkedin": r'https?://(?:www\.)?linkedin\.com/(?:company/|in/|profile/view\?id=)[\w\-]+',
        "twitter": r'https?://(?:www\.)?x\.com/[\w\-]+',
        "facebook": r'https?://(?:www\.)?facebook\.com/[\w\.\-]+',
        "instagram": r'https?://(?:www\.)?instagram\.com/[\w\.\-]+',
        "youtube": r'https?://(?:www\.)?youtube\.com/(?:channel/|user/)[\w\-]+',
        "github": r'https?://(?:www\.)?github\.com/[\w\-]+'
    }
    
    social_profiles = {}
    for platform, pattern in social_patterns.items():
        matches = re.findall(pattern, html)
        if matches:
            social_profiles[platform] = matches[0]
            
    return social_profiles

async def analyze_with_ai(url: str, content: Dict[str, Any], extract_company_info: bool = True, deep_analysis: bool = False) -> Dict[str, Any]:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Basic relevance check
        relevance_prompt = (
            "You're an AI assistant helping with lead generation.\n"
            "Determine if the following webpage contains useful contact info or business lead information.\n"
            f"Page title: {content['title']}\n"
            f"URL: {url}\n"
            f"Content snippet:\n{content['text'][:2000]}...\n\n"
            "First, respond with either 'Yes' or 'No' for whether this is relevant for lead generation.\n"
            "Second, on a scale of 0-100, rate how valuable this lead appears to be (higher = better lead).\n"
            "Format: Yes/No|Score"
        )

        response = await asyncio.to_thread(
            lambda: model.generate_content(relevance_prompt).text.strip()
        )
        
        # Parse response
        relevance_parts = response.split('|')
        is_relevant = "yes" in relevance_parts[0].lower()
        relevance_score = 0.0
        if len(relevance_parts) > 1:
            try:
                relevance_score = float(re.search(r'\d+', relevance_parts[1]).group()) / 100.0
            except (AttributeError, IndexError, ValueError):
                relevance_score = 0.5 if is_relevant else 0.0
                
        result = {
            "relevant": is_relevant,
            "relevance_score": relevance_score
        }
        
        # Extract company info if requested
        if extract_company_info and is_relevant:
            company_prompt = (
                "You're an AI assistant analyzing business websites for lead generation.\n"
                f"URL: {url}\n"
                f"Page title: {content['title']}\n"
                f"Content snippet:\n{content['text'][:3000]}...\n\n"
                "Extract the following information in JSON format:\n"
                "{\n"
                "  \"name\": \"company name or null if uncertain\",\n"
                "  \"industry\": \"company industry or null\",\n"
                "  \"size_estimate\": \"company size estimate or null\",\n"
                "  \"description\": \"brief company description or null\"\n"
                "}"
            )
            
            company_response = await asyncio.to_thread(
                lambda: model.generate_content(company_prompt).text.strip()
            )
            
            # Extract JSON from response
            try:
                json_match = re.search(r'{.*}', company_response, re.DOTALL)
                if json_match:
                    company_info = json.loads(json_match.group())
                    result["org_info"] = company_info
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Failed to parse company info for {url}: {e}")
                result["org_info"] = {
                    "name": None,
                    "industry": None,
                    "size_estimate": None,
                    "description": None
                }
        
        # Perform deep analysis if requested
        if deep_analysis and is_relevant:
            analysis_prompt = (
                "You're an AI assistant performing advanced lead analysis for sales and marketing.\n"
                f"URL: {url}\n"
                f"Page title: {content['title']}\n"
                f"Content snippet:\n{content['text'][:4000]}...\n\n"
                "Provide a concise analysis of this lead, focusing on:\n"
                "1. Key decision makers or contact points\n"
                "2. Potential business needs or pain points\n"
                "3. Recommended approach for engagement\n"
                "4. Any specific opportunities identified\n"
                "Respond in under 200 words."
            )
            
            analysis_response = await asyncio.to_thread(
                lambda: model.generate_content(analysis_prompt).text.strip()
            )
            result["analysis"] = analysis_response
            
        return result
        
    except Exception as e:
        logger.error(f"[AI ANALYSIS ERROR] {url}: {e}")
        return {"relevant": False, "relevance_score": 0.0}

# Scrape Logic
async def process_url(url: str, options: Dict[str, Any]) -> Lead:
    domain = urlparse(url).netloc
    content = await extract_text_from_url(url)
    
    # Extract contact information
    emails = extract_emails(content["text"])
    phones = extract_phones(content["text"]) if options.get("extract_phones", True) else []
    social_media = extract_social_media(content["html"]) if options.get("extract_social_media", True) else {}
    
    # Create contact info
    contact_info = Contact(
        emails=emails,
        phones=phones,
        social_media=social_media
    )
    
    # Analyze with AI
    analysis_results = await analyze_with_ai(
        url, 
        content, 
        extract_company_info=options.get("extract_company_info", True),
        deep_analysis=options.get("deep_analysis", False)
    )
    
    # Extract org info
    org_info = OrganizationInfo(
        name=analysis_results.get("org_info", {}).get("name"),
        industry=analysis_results.get("org_info", {}).get("industry"),
        size_estimate=analysis_results.get("org_info", {}).get("size_estimate"),
        description=analysis_results.get("org_info", {}).get("description")
    )
    
    # Create lead
    return Lead(
        id=str(uuid.uuid4()),
        url=str(url),
        domain=domain,
        title=content["title"],
        contact_info=contact_info,
        org_info=org_info,
        relevant=analysis_results.get("relevant", False),
        relevance_score=analysis_results.get("relevance_score", 0.0),
        analysis=analysis_results.get("analysis"),
        timestamp=datetime.now().isoformat()
    )

async def process_scraping(urls: List[str], options: Dict[str, Any] = None) -> List[Lead]:
    if options is None:
        options = {}
        
    tasks = [process_url(url, options) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing {urls[i]}: {result}")
        else:
            valid_results.append(result)
            
    return valid_results

async def save_results_to_csv(leads: List[Lead], filename: str) -> str:
    filepath = os.path.join(EXPORT_DIR, filename)
    
    # Convert leads to flattened dict format
    flattened_data = []
    for lead in leads:
        lead_dict = lead.model_dump()
        flat_lead = {
            "id": lead_dict["id"],
            "url": lead_dict["url"],
            "domain": lead_dict["domain"],
            "title": lead_dict["title"],
            "relevant": lead_dict["relevant"],
            "relevance_score": lead_dict["relevance_score"],
            "timestamp": lead_dict["timestamp"],
            "emails": ", ".join(lead_dict["contact_info"]["emails"]),
            "phones": ", ".join(lead_dict["contact_info"]["phones"]),
            "company_name": lead_dict["org_info"]["name"],
            "industry": lead_dict["org_info"]["industry"],
            "company_size": lead_dict["org_info"]["size_estimate"],
            "company_description": lead_dict["org_info"]["description"],
            "analysis": lead_dict["analysis"]
        }
        
        # Add social media fields
        for platform, url in lead_dict["contact_info"]["social_media"].items():
            flat_lead[f"{platform}_url"] = url
            
        flattened_data.append(flat_lead)
    
    # Write to CSV
    async with aiofiles.open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Get all possible headers from all leads
        headers = set()
        for lead in flattened_data:
            headers.update(lead.keys())
        
        writer_buffer = io.StringIO()
        writer = csv.DictWriter(writer_buffer, fieldnames=sorted(headers))
        writer.writeheader()
        writer.writerows(flattened_data)
        
        await f.write(writer_buffer.getvalue())
    
    return filepath

# Background processing for batch jobs
async def process_batch_job(job_id: str, urls: List[str], options: Dict[str, Any]):
    try:
        active_jobs[job_id].update({
            "status": "processing",
            "total_urls": len(urls),
            "processed_urls": 0
        })
        
        batch_size = 10
        results = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            batch_results = await process_scraping(batch, options)
            results.extend(batch_results)
            
            # Update progress
            active_jobs[job_id]["processed_urls"] = min(i + batch_size, len(urls))
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"leads_{job_id}_{timestamp}.csv"
        csv_path = await save_results_to_csv(results, filename)
        
        # Update job status
        active_jobs[job_id].update({
            "status": "completed",
            "completed": True,
            "results_file": filename
        })
        
    except Exception as e:
        logger.error(f"Error processing batch job {job_id}: {e}")
        active_jobs[job_id].update({
            "status": "error",
            "error": str(e)
        })

# API Endpoints
@app.post("/scrape", response_model=List[Lead])
async def scrape_and_filter(req: ScrapeRequest, request: Request):
    if not req.urls:
        raise HTTPException(status_code=400, detail="URL list cannot be empty.")
    
    # Extract options from request
    options = {
        "extract_phones": req.extract_phones,
        "extract_social_media": req.extract_social_media,
        "extract_company_info": req.extract_company_info,
        "deep_analysis": req.deep_analysis,
    }
    
    # Process the URLs
    results = await process_scraping([str(url) for url in req.urls], options)
    
    # Store results in app state for later export
    if not hasattr(request.app.state, "current_results"):
        request.app.state.current_results = {}
    
    # Generate a unique session ID for these results
    session_id = str(uuid.uuid4())
    request.app.state.current_results[session_id] = results
    
    # Set a session ID cookie
    response = JSONResponse(content=[lead.model_dump() for lead in results])
    response.set_cookie(key="lead_session", value=session_id, max_age=1800)  # 30 minutes expiry
    
    return response

@app.post("/batch-upload")
async def batch_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_phones: bool = Form(True),
    extract_social_media: bool = Form(True),
    extract_company_info: bool = Form(True),
    deep_analysis: bool = Form(False)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if 'url' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain a 'url' column.")
        
        urls = df['url'].dropna().tolist()
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs found in CSV file.")
        
        job_id = str(uuid.uuid4())
        
        active_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "total_urls": len(urls),
            "processed_urls": 0,
            "completed": False
        }
        
        options = {
            "extract_phones": extract_phones,
            "extract_social_media": extract_social_media,
            "extract_company_info": extract_company_info,
            "deep_analysis": deep_analysis
        }
        
        # Start background task
        background_tasks.add_task(process_batch_job, job_id, urls, options)
        
        return {"job_id": job_id}
    
    except Exception as e:
        logger.error(f"Error processing CSV upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/jobs", response_model=List[JobStatus])
async def get_job_status():
    return [
        JobStatus(
            job_id=job_id,
            status=job_data["status"],
            total_urls=job_data.get("total_urls", 0),
            processed_urls=job_data.get("processed_urls", 0),
            completed=job_data.get("completed", False),
            results_file=job_data.get("results_file")
        )
        for job_id, job_data in active_jobs.items()
    ]

@app.post("/export-current-results")
async def export_current_results(request: Request):
    if not hasattr(request.app.state, "current_results"):
        request.app.state.current_results = {}
    
    session_id = request.cookies.get("lead_session")
    if not session_id or session_id not in request.app.state.current_results:
        raise HTTPException(status_code=404, detail="No results available to export.")
    
    results = request.app.state.current_results[session_id]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"leads_export_{timestamp}.csv"
    csv_path = await save_results_to_csv(results, filename)
    
    return {"filename": os.path.basename(csv_path)}

@app.get("/downloads")
async def list_downloads():
    files = []
    for filename in os.listdir(EXPORT_DIR):
        if filename.endswith('.csv'):
            file_path = os.path.join(EXPORT_DIR, filename)
            stat = os.stat(file_path)
            files.append({
                "name": filename,
                "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                "size": stat.st_size
            })
    
    return files

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(EXPORT_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(path=file_path, filename=filename, media_type='text/csv')

@app.get("/", response_class=HTMLResponse)
async def get_html_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
