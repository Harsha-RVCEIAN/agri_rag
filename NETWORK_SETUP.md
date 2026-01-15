# 🌐 Agri-RAG Network Setup Guide

## 📍 Your Laptop (Backend Server)

**Your IP Address:** `192.168.1.117`

### Step 1: Start the Backend Server

Open PowerShell and run:

```powershell
cd C:\Users\harsh\agri_rag
.\env\Scripts\Activate.ps1
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

✅ Your backend is now accessible from any device on the network at: **http://192.168.1.117:8000**

---

## 👥 Your Friend's Laptop (Frontend Client)

### Step 1: Access the Frontend

1. **On your friend's laptop**, open a web browser
2. Navigate to the frontend (hosted locally or from any accessible server)
3. A configuration modal will appear asking for the **Backend IP Address**

### Step 2: Enter Your Backend IP

In the configuration modal, enter: **`192.168.1.117`**

![Config Modal]
```
⚙️ Backend Configuration
Enter the IP address of your backend server
┌─────────────────────────────────┐
│ 192.168.1.117                   │
└─────────────────────────────────┘
   [Connect to Backend]
```

### Step 3: Connect and Chat

✅ Your friend can now:
- Ask agricultural questions
- Use voice input (will record and send to your backend)
- Get responses from your backend RAG system

---

## 🔌 Network Requirements

### On Your Laptop (Backend)
- ✅ Backend running on `0.0.0.0:8000` (accessible from all interfaces)
- ✅ No firewall blocking port 8000
- ✅ Both on same WiFi network

### On Friend's Laptop (Frontend)
- ✅ Connected to same WiFi network
- ✅ Can ping your IP: `ping 192.168.1.117`
- ✅ Browser can access: `http://192.168.1.117:8000`

---

## 🔒 Firewall Configuration (If Needed)

### Windows Firewall - Allow Port 8000

**Option A: PowerShell (Admin)**
```powershell
New-NetFirewallRule -DisplayName "Agri-RAG Backend" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000
```

**Option B: Manual**
1. Open **Windows Defender Firewall** → **Advanced Settings**
2. Click **Inbound Rules** → **New Rule**
3. Protocol: **TCP**, Local Port: **8000**
4. Allow the connection

---

## 🧪 Test the Connection

### From Your Friend's Laptop

**Method 1: Browser**
```
http://192.168.1.117:8000
```
Should show:
```json
{"status": "ok"}
```

**Method 2: Terminal (PowerShell)**
```powershell
Invoke-WebRequest -Uri "http://192.168.1.117:8000" -Method GET
```

**Method 3: Test Chat Endpoint**
```powershell
$body = @{question="What is drip irrigation?"} | ConvertTo-Json
Invoke-WebRequest -Uri "http://192.168.1.117:8000/chat" `
  -Method POST `
  -Headers @{"Content-Type"="application/json"} `
  -Body $body
```

---

## 📱 Frontend Files to Share with Friend

Share the entire `/frontend` folder:
```
frontend/
├── index.html              ← Main entry point
├── assets/
│   ├── css/               ← Styles
│   ├── js/                ← Chat, voice, API scripts
│   └── images/            ← Logo
```

Friend can:
- Open `index.html` in a browser locally
- Or serve it with: `python -m http.server 8001` and access `http://localhost:8001`

---

## 🐛 Troubleshooting

### ❌ "Connection refused" on friend's laptop

**Check:**
1. Is your backend running? `uvicorn api.app:app --host 0.0.0.0 --port 8000`
2. Is port 8000 open? Run Windows Firewall command above
3. Is the IP correct? Your IP: **192.168.1.117**
4. Are you on the same network?

**Fix:**
```powershell
# On your laptop, test locally
curl http://localhost:8000

# On friend's laptop, test your IP
# Windows PowerShell
Test-NetConnection -ComputerName 192.168.1.117 -Port 8000
```

### ❌ "Empty question" error

Friend needs to enter a question in the text box before clicking Send.

### ❌ Voice not working

1. Check if friend's browser allows microphone access
2. Verify audio file is being sent to `/chat/voice` endpoint
3. Check backend logs for STT errors

---

## 📊 API Endpoints (For Reference)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/chat` | POST | Text question → Answer |
| `/chat/voice` | POST | Audio file → Transcribed text |
| `/upload-pdf` | POST | Upload PDF documents |

### Example: Text Query
```json
POST http://192.168.1.117:8000/chat
Content-Type: application/json

{
  "question": "What is the best fertilizer for rice?"
}
```

### Example: Voice Query
```
POST http://192.168.1.117:8000/chat/voice
Content-Type: multipart/form-data

file: <audio.wav>
```

---

## ✅ Success Checklist

- [ ] Backend running on your laptop at `192.168.1.117:8000`
- [ ] Port 8000 is open in firewall
- [ ] Friend's laptop can ping `192.168.1.117`
- [ ] Friend opens frontend and enters IP `192.168.1.117`
- [ ] Friend can ask agricultural questions
- [ ] Friend receives answers from your backend

---

**Need Help?** Check the backend logs on your laptop for errors! 🚀
