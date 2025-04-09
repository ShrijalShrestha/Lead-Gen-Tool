# Advanced Lead Generation Tool

An AI-powered web application that helps businesses extract and analyze potential leads from websites.

## Features

- **URL Analysis**: Analyze individual or multiple URLs for potential leads
- **Batch Processing**: Upload CSV files containing URLs for bulk processing
- **Comprehensive Data Extraction**:
  - Contact information (emails, phone numbers)
  - Social media profiles
  - Company information
  - AI-powered relevance scoring
- **Deep Analysis**: Optional AI-based analysis of companies, their needs, and engagement strategies
- **Export Functionality**: Download results in CSV format
- **Job Management**: Track the progress of batch jobs
- **Modern Web Interface**: User-friendly interface with tabbed navigation

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (sign up at [Google AI Studio](https://ai.google.dev/))

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd LeadGenTool
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

### Running the Application

1. Start the application:
   ```
   python main.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

## Usage Guide

### Analyzing Individual URLs

1. Navigate to the "Single/Multiple URLs" tab
2. Enter one or more URLs (one per line)
3. Select the extraction options
4. Click "Analyze URLs"
5. View and download the results

### Batch Processing

1. Navigate to the "Batch Upload" tab
2. Prepare a CSV file with a column named "url" containing the URLs to analyze
3. Upload the CSV file
4. Select the extraction options
5. Click "Start Batch Processing"
6. Track the job status in the "Active Jobs" tab
7. Download the results when the job is complete

### Viewing and Downloading Results

- Use the "Downloads" tab to access all previously generated exports
- Click "Download" to get the CSV file for any export

## Project Structure

- `main.py`: The main application code
- `requirements.txt`: List of Python dependencies
- `static/styles.css`: CSS styles for the web interface
- `templates/index.html`: HTML template for the web interface
- `exports/`: Directory where CSV exports are stored
- `.env`: Environment variables (API keys)

## API Endpoints

The application provides the following API endpoints:

- `GET /`: Main web interface
- `POST /scrape`: Analyze URLs for leads
- `POST /batch-upload`: Upload and process a CSV file of URLs
- `GET /jobs`: Get status of all jobs
- `POST /export-current-results`: Export current results to CSV
- `GET /downloads`: List available downloads
- `GET /download/{filename}`: Download a specific file

## Technical Implementation

The tool is built with:

- **FastAPI**: High-performance web framework
- **Google Gemini AI**: For advanced analysis and information extraction
- **BeautifulSoup**: For web scraping
- **Pandas**: For data processing
- **Jinja2**: For HTML templating
- **JavaScript**: For frontend interactivity

## License

[Include license information here]

## Contributing

[Include contribution guidelines here if applicable]