# Virtual Environment Setup for ND2 Threshold Analysis

This guide will help you set up a virtual environment and test the new interactive threshold analysis feature.

## 🐍 Step 1: Create Virtual Environment

Open PowerShell and navigate to your project directory:

```powershell
# Navigate to your project
cd "C:\Users\Duke\Downloads\nd2-analysis-pipeline"

# Create virtual environment
python -m venv venv_threshold

# Activate the virtual environment
venv_threshold\Scripts\activate

# You should see (venv_threshold) in your prompt
```

## 📦 Step 2: Install Dependencies

Install all required packages:

```powershell
# Install main pipeline dependencies
pip install -r requirements.txt

# Install additional threshold analysis dependencies
pip install -r threshold_analysis/requirements.txt

# Verify installation
pip list | findstr fastapi
pip list | findstr numpy
```

## 🧪 Step 3: Test the Threshold Analysis

Now test with your actual file:

```powershell
# Test command (replace with your actual file path)
python test_threshold_analysis.py "\\nas.uic.umn.edu\user\User Data Goes Here\Julia Nguyen\Study 2 Octapharrma Lung 20X vWF Pselectin CdD31 05072025\SS Untreated\Study 2 Set #1  Octapharma 20X HbSS Untreated Y3   VWF(R) Psel (G) CD31 (B).nd2" "examples/configs/C1inh_lung_study2.json"
```

## 🚀 Step 4: Test the Web Interface (Optional)

If the threshold analysis works, you can test the web interface:

### Start the API Server:
```powershell
# In terminal 1 (keep virtual environment active)
cd threshold_analysis
python -m web_api.main

# Should show: "Uvicorn running on http://127.0.0.1:8000"
```

### Start the React Frontend:
```powershell
# In terminal 2 (new PowerShell window)
cd "C:\Users\Duke\Downloads\nd2-analysis-pipeline\threshold_analysis\web_interface"

# Install Node.js dependencies
npm install

# Start the development server
npm start

# Should open browser at http://localhost:3000
```

## 🔧 Troubleshooting

### Issue 1: Python not found
```powershell
# Check Python installation
python --version

# If not found, install Python 3.8+ from python.org
```

### Issue 2: Module import errors
```powershell
# Make sure you're in the right directory
pwd
# Should show: C:\Users\Duke\Downloads\nd2-analysis-pipeline

# Make sure virtual environment is active
# You should see (venv_threshold) in prompt
```

### Issue 3: ND2 file access issues
```powershell
# Test if you can access the file
dir "\\nas.uic.umn.edu\user\User Data Goes Here\Julia Nguyen\Study 2 Octapharrma Lung 20X vWF Pselectin CdD31 05072025\SS Untreated\"

# If network drive issues, try copying file locally first
```

### Issue 4: Missing dependencies
```powershell
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
pip install --upgrade -r threshold_analysis/requirements.txt
```

## 📊 What Should Happen

### Successful Test Output:
```
🔬 Testing Threshold Analysis
==================================================
ND2 File: [your file path]
Config: examples/configs/C1inh_lung_study2.json

📊 Processing file with all thresholds (0-4095)...
Success! Processed [filename]
Mouse: Y3, Group: HbSS Untreated
Channel 1 at threshold 1000: 15.23%
Channel 2 at threshold 1000: 8.45%
Channel 3 at threshold 1000: 25.67%

✅ SUCCESS: Threshold analysis completed!
```

### Web Interface:
- Three sliders (Channel 1, 2, 3)
- Default values: 2500, 2500, 300 (from your config)
- Boxplot showing mock data
- Sliders respond smoothly

## 🎯 Next Steps After Successful Test

1. **Generate real threshold data** for multiple files
2. **Connect real data** to the web interface  
3. **Add more visualization options**
4. **Deploy** for production use

## 💡 Why Virtual Environment?

- **Isolation**: Doesn't affect your system Python
- **Reproducibility**: Same environment on different machines
- **Dependency management**: Clean install/uninstall
- **Testing**: Safe to experiment without breaking existing code

## 🔄 Deactivate Virtual Environment

When you're done testing:

```powershell
# Deactivate the virtual environment
deactivate

# The (venv_threshold) prefix should disappear
```


