# Multi-Asset Trading Dashboard

A modern, responsive trading dashboard built with React, TypeScript, and Vite. Features real-time stock predictions, portfolio management, market scanning, and comprehensive analytics.

## 🚀 Features

- **🔐 Authentication**: JWT-based authentication with secure session management
- **📊 Dashboard**: Real-time portfolio performance and market overview
- **🔍 Market Scan**: AI-powered stock analysis and prediction engine
- **💼 Portfolio**: Backend-authoritative portfolio management with real-time updates
- **📈 Analytics**: Advanced charting and performance metrics
- **⭐ Watch List**: Custom stock monitoring with alerts
- **🔔 Alerts**: Price and performance-based notification system
- **🤖 AI Assistant**: Intelligent trading insights and recommendations
- **🛡️ Risk Management**: Stop-loss and position sizing tools

## 🏗️ Architecture

### Core Principle: Backend Authority

**Frontend does not interpret backend dependency state.**

Backend errors are rendered verbatim. Any deviation is a bug.

The backend is the single source of truth for all data, state, and errors. The frontend:
- Asks (makes API requests)
- Waits (shows neutral loading)
- Renders (displays backend responses exactly)
- Fails (shows backend errors verbatim)

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed principles.

### Frontend (React 19 + TypeScript + Vite)
- **Backend-Authoritative Design**: All financial data sourced exclusively from backend
- **No Client-Side Calculations**: Portfolio values, PnL, and totals calculated server-side
- **Error-First Approach**: Shows errors instead of fallback data when backend unavailable
- **Real-Time Updates**: Polling-based data refresh with intelligent rate limiting

### Backend (FastAPI + Python)
- **Secure API**: JWT authentication with rate limiting and validation
- **ML-Powered Predictions**: Ensemble models for stock price forecasting
- **Risk Engine**: Advanced risk assessment and management tools
- **Data Validation**: Strict input validation and error handling

## 🛠️ Technology Stack

### Frontend
- **React 19** - Modern UI library
- **TypeScript** - Type-safe development
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first styling
- **Axios** - HTTP client
- **React Router** - Client-side routing

### Backend
- **FastAPI** - High-performance Python web framework
- **PyTorch** - Machine learning framework
- **scikit-learn** - Traditional ML algorithms
- **pandas** - Data manipulation and analysis
- **JWT** - Secure authentication
- **Uvicorn** - ASGI server

## 🚀 Quick Start

### Prerequisites
- Node.js 16+
- Python 3.8+
- npm or yarn

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python api_server.py
```

### Frontend Setup
```bash
cd trading-dashboard
npm install
npm run dev
```

### Environment Configuration
Create `.env` file in backend directory:
```bash
ENABLE_AUTH=false
JWT_SECRET_KEY=your-secret-key
UVICORN_HOST=127.0.0.1
UVICORN_PORT=8000
DEBUG_MODE=true
```

## 🔧 Key Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/logout` - User logout
- `GET /auth/status` - Authentication status

### Trading Tools
- `POST /tools/predict` - Stock price predictions
- `POST /tools/scan_all` - Market scanning and ranking
- `POST /tools/analyze` - Detailed stock analysis
- `POST /tools/feedback` - User feedback collection

### Risk Management
- `POST /api/risk/assess` - Risk assessment
- `POST /api/risk/stop-loss` - Stop-loss management

### Portfolio
- `GET /api/portfolio` - Portfolio data (backend-calculated)
- `POST /api/portfolio/add` - Add position
- `POST /api/portfolio/remove` - Remove position

## 📊 Data Flow

1. **Frontend requests** financial data from backend
2. **Backend calculates** all values server-side (no client math)
3. **Backend returns** structured data with pre-calculated totals
4. **Frontend validates** data integrity before rendering
5. **Error handling** shows clear messages when backend unavailable

## 🔒 Security

- **JWT Authentication**: Secure token-based authentication
- **Rate Limiting**: Prevents API abuse (500 requests/minute)
- **Input Validation**: Strict data validation on all endpoints
- **CORS Protection**: Controlled cross-origin resource sharing
- **No Client Calculations**: Prevents manipulation of financial data

## 🎯 Integration Status

### Fully Integrated (Backend-Authoritative)
- ✅ MarketScanPage.tsx
- ✅ AnalyticsPage.tsx

### Partially Integrated (Hybrid)
- ⚠️ DashboardPage.tsx
- ⚠️ PortfolioPage.tsx
- ⚠️ WatchListPage.tsx
- ⚠️ AlertsPage.tsx
- ⚠️ ComparePage.tsx
- ⚠️ TradingHistoryPage.tsx
- ⚠️ SettingsPage.tsx
- ⚠️ TrainModelPage.tsx
- ⚠️ LoginPage.tsx

### Not Integrated
- ❌ SignupPage.tsx
- ❌ UserProfilePage.tsx
- ❌ EducationalDashboardPage.tsx

## 📈 Performance

- **Real-time Updates**: 2-minute polling intervals
- **Rate Limiting**: 500 requests/minute, 10,000/hour
- **Caching**: Intelligent caching for non-critical data
- **Error Recovery**: Automatic retry logic with exponential backoff

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure backend server is running on `http://127.0.0.1:8000`
   - Check firewall settings
   - Verify CORS configuration

2. **Authentication Errors**
   - Check JWT configuration
   - Verify token expiration settings
   - Ensure proper header formatting

3. **Rate Limit Exceeded**
   - Reduce polling frequency
   - Check `config.py` rate limit settings
   - Monitor API usage patterns

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

For issues and questions:
- Open GitHub issues
- Check documentation
- Review existing troubleshooting guides