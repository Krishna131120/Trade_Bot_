# Quick Start Guide - Frontend ↔ Backend Connection

## ✅ Configuration Status: READY

Your frontend is already configured to connect to localhost backend!

---

## 🚀 Start the Application

### Step 1: Start Backend Server

```bash
cd backend
python api_server.py
```

**Expected Output:**
```
================================================================================
                    MCP API SERVER STARTING
================================================================================

Server starting on http://0.0.0.0:8000
```

**Backend will be available at:**
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/tools/health

---

### Step 2: Start Frontend (New Terminal)

```bash
cd trading-dashboard
npm run dev
```

**Expected Output:**
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

**Frontend will be available at:**
- App: http://localhost:5173

---

## 🔧 Current Configuration

### Backend (`backend/config.py`)
```python
UVICORN_HOST = '0.0.0.0'  # Accepts connections from anywhere
UVICORN_PORT = 8000        # Backend port
```

### Frontend (`trading-dashboard/.env`)
```env
VITE_API_BASE_URL=http://127.0.0.1:8000  # ✅ Points to localhost backend
VITE_ENABLE_AUTH=false                    # ✅ Auth disabled (matches backend)
```

### Frontend Config (`trading-dashboard/src/config.ts`)
```typescript
API_BASE_URL: 'http://127.0.0.1:8000'  // ✅ Localhost backend
```

---

## ✅ Verification Steps

### 1. Test Backend (Before Starting Frontend)

Open browser: http://localhost:8000

**Expected Response:**
```json
{
  "name": "Stock Prediction MCP API",
  "version": "4.0",
  "authentication": "DISABLED - Open access to all endpoints",
  "endpoints": {
    "/tools/predict": "POST - Generate predictions",
    ...
  }
}
```

### 2. Test Frontend Connection

1. Open frontend: http://localhost:5173
2. Open browser console (F12)
3. Click any stock tab (e.g., TCS, RELIANCE)
4. Check console logs:

```
[TAB] Clicked: TCS.NS
[API] /tools/predict will be called for TCS.NS
[API] POST /tools/predict called for TCS.NS
[API] ✅ Success - prediction generated for TCS.NS
[RENDER] Success card: TCS.NS
```

---

## 🔍 Troubleshooting

### Issue: Frontend can't connect to backend

**Check 1: Is backend running?**
```bash
# Windows
netstat -ano | findstr :8000

# Should show: TCP 0.0.0.0:8000 ... LISTENING
```

**Check 2: Test backend directly**
```bash
curl http://localhost:8000
```

**Check 3: Check frontend config**
- Open: `trading-dashboard/.env`
- Verify: `VITE_API_BASE_URL=http://127.0.0.1:8000`

**Check 4: Restart frontend after .env changes**
```bash
# Stop frontend (Ctrl+C)
# Start again
npm run dev
```

---

### Issue: CORS errors

**Solution:** Backend already has CORS enabled for all origins:
```python
# In api_server.py (already configured)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Issue: Port 8000 already in use

**Solution 1: Kill existing process**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID_NUMBER>
```

**Solution 2: Change backend port**
1. Edit `backend/.env` (create if doesn't exist):
   ```env
   UVICORN_PORT=8001
   ```
2. Edit `trading-dashboard/.env`:
   ```env
   VITE_API_BASE_URL=http://127.0.0.1:8001
   ```
3. Restart both servers

---

## 📊 Connection Flow

```
User Browser (localhost:5173)
        ↓
Frontend React App
        ↓
API Call: http://127.0.0.1:8000/tools/predict
        ↓
Backend FastAPI Server (localhost:8000)
        ↓
MCPAdapter → stock_analysis_complete.py
        ↓
ML Models (Random Forest, XGBoost, LightGBM, DQN)
        ↓
Response → Frontend → User
```

---

## 🎯 Quick Test

### Test Prediction Endpoint

**Using curl:**
```bash
curl -X POST http://localhost:8000/tools/predict \
  -H "Content-Type: application/json" \
  -d "{\"symbols\": [\"TCS.NS\"], \"horizon\": \"intraday\"}"
```

**Using browser console (when frontend is open):**
```javascript
fetch('http://localhost:8000/tools/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ symbols: ['TCS.NS'], horizon: 'intraday' })
})
.then(r => r.json())
.then(console.log)
```

---

## ✅ Success Indicators

### Backend Running Successfully:
- ✅ Console shows: "Server starting on http://0.0.0.0:8000"
- ✅ http://localhost:8000 returns API info
- ✅ http://localhost:8000/docs shows Swagger UI

### Frontend Connected Successfully:
- ✅ Frontend loads at http://localhost:5173
- ✅ No CORS errors in browser console
- ✅ Stock tabs show predictions or "unavailable" cards
- ✅ Console shows `[API] POST /tools/predict called for SYMBOL`

---

## 📝 Notes

- **Backend must start BEFORE frontend** for initial connection
- **Frontend auto-reconnects** if backend restarts
- **Dev logs only show in development mode** (npm run dev)
- **Backend logs** are in `backend/data/logs/api_server.log`
- **Frontend uses port 5173** by default (Vite)
- **Backend uses port 8000** by default (FastAPI)

---

## 🎉 You're All Set!

Your configuration is correct. Just start both servers and you're ready to go!

1. Terminal 1: `cd backend && python api_server.py`
2. Terminal 2: `cd trading-dashboard && npm run dev`
3. Browser: http://localhost:5173

Happy Trading! 📈
