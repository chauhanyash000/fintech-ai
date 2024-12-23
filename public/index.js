const functions = require('firebase-functions');
const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

const app = express();

// Middleware
app.use(cors({ origin: true }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Store report status and report builders
const reportStatus = {};
const reportBuilders = {};

// Helper function to run Python script
async function runPythonScript(script, args) {
  return new Promise((resolve, reject) => {
    const pythonProcess = exec(`python ${script} ${args}`, {
      env: {
        ...process.env,
        OPENAI_API_KEY: process.env.OPENAI_API_KEY,
        GOOGLE_API_KEY: process.env.GOOGLE_API_KEY,
        CUSTOM_ID: process.env.CUSTOM_ID
      }
    });

    let output = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      output += data;
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data;
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script failed with error: ${error}`));
      } else {
        resolve(output);
      }
    });
  });
}

// Routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/generate', async (req, res) => {
  const ticker = req.body.ticker.toUpperCase();
  const companyName = req.body.company_name;

  if (!ticker || !companyName) {
    return res.status(400).json({ error: 'Please provide both ticker and company name' });
  }

  if (reportStatus[ticker] && reportStatus[ticker] !== 'completed' && reportStatus[ticker] !== 'failed') {
    return res.status(409).json({ error: 'Report generation already in progress' });
  }

  reportStatus[ticker] = 'starting';

  // Create temporary directory for analysis
  const tempDir = path.join(os.tmpdir(), `analysis_pdfs_${ticker}`);
  await fs.mkdir(tempDir, { recursive: true });

  // Start analysis pipeline in background
  runAnalysisPipeline(ticker, companyName, tempDir)
    .catch(error => {
      console.error(`Error in analysis pipeline: ${error}`);
      reportStatus[ticker] = 'failed';
    });

  res.json({ status: 'starting', ticker });
});

app.get('/status/:ticker', (req, res) => {
  const ticker = req.params.ticker.toUpperCase();
  const status = reportStatus[ticker] || 'not_found';
  
  const messages = {
    'starting': 'Starting analysis pipeline...',
    'running_ctf': 'Analyzing technical and fundamental data...',
    'running_key_takeaways': 'Generating key takeaways...',
    'running_analyst_coverage': 'Analyzing analyst coverage...',
    'running_earnings': 'Processing earnings call...',
    'running_press_release': 'Analyzing press releases...',
    'running_price_changes': 'Analyzing price changes...',
    'running_research': 'Gathering research articles...',
    'combining_reports': 'Generating professional report...',
    'completed': 'Analysis complete',
    'failed': 'Analysis failed',
    'not_found': 'Analysis not found'
  };

  res.json({ 
    status, 
    message: messages[status] || 'Unknown status' 
  });
});

app.get('/download/:ticker', async (req, res) => {
  const ticker = req.params.ticker.toUpperCase();
  const filename = `${ticker}_Investment_Analysis.pdf`;
  const filePath = path.join(os.tmpdir(), filename);

  try {
    await fs.access(filePath);
    res.download(filePath, filename);
  } catch (error) {
    res.status(404).json({ error: 'Report not found' });
  }
});

app.post('/cleanup/:ticker', async (req, res) => {
  const ticker = req.params.ticker.toUpperCase();
  const filename = `${ticker}_Investment_Analysis.pdf`;
  const filePath = path.join(os.tmpdir(), filename);

  try {
    await fs.unlink(filePath);
    delete reportStatus[ticker];
    res.json({ message: 'Cleanup successful' });
  } catch (error) {
    res.status(500).json({ error: `Cleanup failed: ${error.message}` });
  }
});

// Firebase Cloud Functions export
exports.app = functions.https.onRequest(app);
