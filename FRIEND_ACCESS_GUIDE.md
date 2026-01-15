# 🌐 How Your Friend Can Access the App

## **Step 1: Get Your Laptop's IP Address**

Run this on your laptop:
```powershell
ipconfig | findstr IPv4
```

**You'll see something like:**
```
IPv4 Address. . . . . . . . . . . : 192.168.1.117
```

**Your IP: `192.168.1.117`** ← Share this with your friend

---

## **Step 2: Stop Current Servers**

Press `CTRL+C` in both terminal windows to stop:
- ❌ Frontend (http.server on port 8080)
- ❌ Backend (uvicorn on port 5000)

---

## **Step 3: Restart Backend on Network Interface**

Run this command on your laptop:

```powershell
cd C:\Users\harsh\agri_rag
.\env\Scripts\Activate.ps1
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
```

✅ Backend now accessible at: `http://192.168.1.117:5000`

---

## **Step 4: Restart Frontend on Network Interface**

Open a **new terminal** and run:

```powershell
cd C:\Users\harsh\agri_rag\frontend
python -m http.server 8080 --bind 0.0.0.0
```

✅ Frontend now accessible at: `http://192.168.1.117:8080`

---

## **For Your Friend's Laptop:**

### **Requirements:**
- ✅ Same WiFi/LAN as your laptop
- ✅ Can reach `192.168.1.117`

### **Step 1: Test Connection**

On friend's laptop, open PowerShell:
```powershell
Test-NetConnection -ComputerName 192.168.1.117 -Port 5000
```

Should show: `TcpTestSucceeded : True`

### **Step 2: Open Frontend**

Friend opens browser and goes to:
```
http://192.168.1.117:8080
```

### **Step 3: Configure Backend IP**

When the configuration modal appears:
- **Enter:** `192.168.1.117`
- **Click:** "Connect to Backend"
- **Done!** Friend can now use the app

---

## **Quick Reference**

| Item | Your Laptop | Friend's Laptop |
|------|------------|-----------------|
| **Frontend URL** | http://localhost:8080 | http://192.168.1.117:8080 |
| **Backend URL** | http://localhost:5000 | http://192.168.1.117:5000 |
| **Backend IP to Enter** | (auto-detects) | **192.168.1.117** |
| **Port** | 5000 | 5000 |

---

## **Troubleshooting**

### ❌ Friend can't reach the backend

**Check:**
1. Run `ipconfig` on your laptop → Confirm IP is `192.168.1.117`
2. Friend tries: `http://192.168.1.117:5000` in browser
3. Should see: `{"status": "ok"}`

**If still failing:**
- Make sure both laptops on same WiFi
- Allow Python through firewall (or restart router)
- Disable VPN on both laptops

### ❌ Frontend loads but can't chat

1. Friend should see config modal
2. Enter `192.168.1.117` 
3. Click "Connect to Backend"
4. Check browser console (F12) for errors

---

## **Commands Summary**

**On YOUR Laptop (2 terminals):**

Terminal 1 - Backend:
```powershell
cd C:\Users\harsh\agri_rag
.\env\Scripts\Activate.ps1
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
```

Terminal 2 - Frontend:
```powershell
cd C:\Users\harsh\agri_rag\frontend
python -m http.server 8080 --bind 0.0.0.0
```

**Share with Friend:**
```
📍 Frontend: http://192.168.1.117:8080
🔌 Backend IP: 192.168.1.117
```

---

**That's it! Your friend can now access the app!** 🚀
