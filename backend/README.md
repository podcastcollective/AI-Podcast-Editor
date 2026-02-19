[README.md](https://github.com/user-attachments/files/25413475/README.md)
# AI Podcast Editor - Backend API

Flask-based REST API for processing podcast episodes with AI agents.

## Quick Deploy to Railway

### Step 1: Push to GitHub

1. In your existing `AI-Podcast-Editor` repo, create a `backend` folder
2. Upload these files to the `backend` folder:
   - `app.py`
   - `requirements.txt`
   - `Procfile`
   - `.gitignore`
   - `README.md`

### Step 2: Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose `AI-Podcast-Editor`
5. Railway will auto-detect the Python app

### Step 3: Add Environment Variables

In Railway dashboard:
1. Click on your deployment
2. Go to **Variables** tab
3. Add these variables:
   ```
   ASSEMBLYAI_API_KEY=your_assemblyai_key_here
   CLAUDE_API_KEY=your_claude_key_here
   ```

### Step 4: Get Your API URL

1. Go to **Settings** tab
2. Under **Domains**, click **Generate Domain**
3. Copy the URL (e.g., `https://ai-podcast-editor.up.railway.app`)
4. This is your API endpoint!

## API Endpoints

### Health Check
```
GET /
```
Returns API status

### Check Configuration
```
GET /api/status
```
Returns configuration status and API key validation

### Upload File
```
POST /api/upload
```
Body: `multipart/form-data` with `file` field
Returns: `{ "success": true, "filename": "..." }`

### Process Podcast
```
POST /api/process
```
Body:
```json
{
  "filename": "uploaded_file.mp3",
  "requirements": {
    "removeFillerWords": true,
    "removeLongPauses": true,
    "normalizeAudio": true,
    "removeBackgroundNoise": true,
    "targetLength": "45 minutes"
  },
  "customInstructions": "Keep the intro, remove technical issues..."
}
```
Returns: Edit decisions and transcript analysis

### Download Files
```
GET /api/download/<job_id>/transcript
GET /api/download/<job_id>/decisions
GET /api/download/<job_id>/report
```

## Local Development

### Setup
```bash
cd backend
pip install -r requirements.txt
```

### Set Environment Variables
```bash
export ASSEMBLYAI_API_KEY="your_key"
export CLAUDE_API_KEY="your_key"
```

### Run
```bash
python app.py
```

Server runs on `http://localhost:5000`

## Testing the API

### Using curl:
```bash
# Health check
curl https://your-api.railway.app/

# Check status
curl https://your-api.railway.app/api/status

# Upload file
curl -X POST -F "file=@podcast.mp3" https://your-api.railway.app/api/upload

# Process
curl -X POST https://your-api.railway.app/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "20250219_120000_podcast.mp3",
    "requirements": {"removeFillerWords": true},
    "customInstructions": "Keep it natural"
  }'
```

## Connecting to Frontend

Update your frontend to use the Railway URL:

```javascript
const API_URL = 'https://your-app.up.railway.app';

// Upload file
const formData = new FormData();
formData.append('file', audioFile);
const uploadResponse = await fetch(`${API_URL}/api/upload`, {
  method: 'POST',
  body: formData
});

// Process
const processResponse = await fetch(`${API_URL}/api/process`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    filename: uploadedFilename,
    requirements: requirements,
    customInstructions: customInstructions
  })
});
```

## Cost Estimates

Per episode (45 min):
- AssemblyAI: $0.19
- Claude API: $0.15
- Railway hosting: ~$5/month (unlimited episodes)

**Total: $0.34 per episode + $5/month hosting**

## Troubleshooting

**"API keys not configured"**
- Make sure you've added ASSEMBLYAI_API_KEY and CLAUDE_API_KEY in Railway Variables

**"File upload fails"**
- Check file size < 500MB
- Ensure file type is MP3, WAV, M4A, AAC, or OGG

**"CORS errors"**
- Make sure Flask-CORS is installed
- Check that your frontend URL is calling the correct Railway URL

## Support

- Railway docs: https://docs.railway.app/
- AssemblyAI docs: https://www.assemblyai.com/docs
- Anthropic docs: https://docs.anthropic.com/
