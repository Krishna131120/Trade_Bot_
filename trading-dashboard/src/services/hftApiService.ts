import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
    HftBotData,
    HftLiveStatus,
    HftMcpStatus,
    HftMcpAnalysisRequest,
    HftMcpAnalysisResponse,
    HftMcpChatRequest,
    HftMcpChatResponse,
    HftWatchlistResponse,
    HftSettingsUpdate,
    HftTrade,
    HftPortfolio
} from '../types/hft';

// Use same backend as health check so BOT and "System Offline" use one server
import { config } from '../config';
const API_BASE_URL = config.API_BASE_URL;

// Create axios instance with default config
const api: AxiosInstance = axios.create({
    baseURL: `${API_BASE_URL}/api`,
    timeout: 60000, // 60 seconds for bot initialization
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor for logging
api.interceptors.request.use(
    (config) => {
        console.log(`[HFT API] Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
    },
    (error: AxiosError) => {
        console.error('[HFT API] Request Error:', error);
        return Promise.reject(error);
    }
);

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => {
        console.log(`[HFT API] Response: ${response.status} ${response.config.url}`);
        return response;
    },
    (error: AxiosError) => {
        console.error('[HFT API] Response Error:', error.response?.data || error.message);

        // Handle specific error cases
        if (error.response?.status === 404) {
            throw new Error('API endpoint not found');
        } else if (error.response?.status === 500) {
            throw new Error('Server error occurred');
        } else if (error.code === 'ECONNABORTED') {
            throw new Error('Request timeout');
        } else if (!error.response) {
            throw new Error('Network error - please check if the backend server is running');
        }

        throw error;
    }
);

export const hftApiService = {
    // ===== Bot Status =====
    async getStatus(): Promise<any> {
        try {
            const response = await api.get('/status');
            return response.data;
        } catch (error) {
            console.error('Error getting bot status:', error);
            throw error;
        }
    },

    // ===== Complete Bot Data =====
    async getBotData(): Promise<HftBotData> {
        try {
            const response = await api.get<HftBotData>('/bot-data', { timeout: 25000 }); // 25s for live Dhan fetch when bot not initialized
            return response.data;
        } catch (error) {
            console.error('Error getting bot data:', error);
            throw error;
        }
    },

    // ===== Portfolio Management =====
    async getPortfolio(): Promise<HftPortfolio> {
        try {
            const response = await api.get<HftPortfolio>('/portfolio');
            return response.data;
        } catch (error) {
            console.error('Error getting portfolio:', error);
            throw error;
        }
    },

    // ===== Trading History =====
    async getTrades(limit: number = 10): Promise<HftTrade[]> {
        try {
            const response = await api.get<HftTrade[]>(`/trades?limit=${limit}`);
            return response.data;
        } catch (error) {
            console.error('Error getting trades:', error);
            throw error;
        }
    },

    // ===== Watchlist Management =====
    async getWatchlist(): Promise<string[]> {
        try {
            const response = await api.get<string[]>('/watchlist');
            // Backend returns array directly, not wrapped in {tickers: []}
            return Array.isArray(response.data) ? response.data : [];
        } catch (error) {
            console.error('Error getting watchlist:', error);
            throw error;
        }
    },

    async addToWatchlist(ticker: string): Promise<HftWatchlistResponse> {
        try {
            const response = await api.post<HftWatchlistResponse>(`/watchlist/add/${ticker}`);
            return response.data;
        } catch (error) {
            console.error('Error adding to watchlist:', error);
            throw error;
        }
    },

    async removeFromWatchlist(ticker: string): Promise<HftWatchlistResponse> {
        try {
            const response = await api.delete<HftWatchlistResponse>(`/watchlist/remove/${ticker}`);
            return response.data;
        } catch (error) {
            console.error('Error removing from watchlist:', error);
            throw error;
        }
    },

    async bulkUpdateWatchlist(tickers: string[], action: 'ADD' | 'REMOVE' = 'ADD'): Promise<HftWatchlistResponse> {
        try {
            const response = await api.post<HftWatchlistResponse>('/watchlist/bulk', {
                tickers,
                action,
            });
            return response.data;
        } catch (error) {
            console.error('Error bulk updating watchlist:', error);
            throw error;
        }
    },

    // ===== Chat/Commands =====
    async sendChatMessage(message: string): Promise<{ response: string; timestamp: string }> {
        try {
            const response = await api.post<{ response: string; timestamp: string }>('/chat', {
                message,
            });
            return response.data;
        } catch (error) {
            console.error('Error sending chat message:', error);
            throw error;
        }
    },

    // ===== MCP (Model Context Protocol) API Endpoints =====
    async mcpAnalyzeMarket(analysisRequest: HftMcpAnalysisRequest): Promise<HftMcpAnalysisResponse> {
        try {
            const response = await api.post<HftMcpAnalysisResponse>('/mcp/analyze', analysisRequest);
            return response.data;
        } catch (error) {
            console.error('MCP market analysis error:', error);
            throw error;
        }
    },

    async mcpExecuteTrade(tradeRequest: any): Promise<any> {
        try {
            const response = await api.post('/mcp/execute', tradeRequest);
            return response.data;
        } catch (error) {
            console.error('MCP trade execution error:', error);
            throw error;
        }
    },

    async mcpChat(chatRequest: HftMcpChatRequest): Promise<HftMcpChatResponse> {
        try {
            const response = await api.post<HftMcpChatResponse>('/mcp/chat', chatRequest);
            return response.data;
        } catch (error) {
            console.error('MCP chat error:', error);
            throw error;
        }
    },

    async getMcpStatus(): Promise<HftMcpStatus> {
        try {
            const response = await api.get<HftMcpStatus>('/mcp/status');
            return response.data;
        } catch (error) {
            console.error('MCP status error:', error);
            throw error;
        }
    },

    // ===== Bot Control =====
    async startBot(): Promise<{ message: string }> {
        try {
            const response = await api.post<{ message: string }>('/bot/start');
            return response.data;
        } catch (error) {
            console.error('Error starting bot:', error);
            throw error;
        }
    },

    async startBotWithSymbol(symbol: string): Promise<any> {
        try {
            // Use longer timeout for bot initialization
            const response = await api.post('/bot/start-with-symbol', { symbol }, { timeout: 10000 }); // 10 seconds - should return quickly
            return response.data;
        } catch (error) {
            console.error('Error starting bot with symbol:', error);
            throw error;
        }
    },

    async stopBot(): Promise<{ message: string }> {
        try {
            const response = await api.post<{ message: string }>('/bot/stop');
            return response.data;
        } catch (error) {
            console.error('Error stopping bot:', error);
            throw error;
        }
    },

    // ===== Settings Management =====
    async getSettings(): Promise<any> {
        try {
            const response = await api.get('/settings');
            return response.data;
        } catch (error) {
            console.error('Error getting settings:', error);
            throw error;
        }
    },

    async updateSettings(settings: HftSettingsUpdate): Promise<{ message: string }> {
        try {
            const response = await api.post<{ message: string }>('/settings', settings);
            return response.data;
        } catch (error) {
            console.error('Error updating settings:', error);
            throw error;
        }
    },

    // ===== Live Trading Status =====
    async getLiveStatus(): Promise<HftLiveStatus> {
        try {
            const response = await api.get<HftLiveStatus>('/live-status', { timeout: 10000 }); // 10 seconds for live status
            return response.data;
        } catch (error) {
            console.error('Error getting live status:', error);
            throw error;
        }
    },

    async syncLivePortfolio(): Promise<any> {
        try {
            const response = await api.post('/live/sync');
            return response.data;
        } catch (error) {
            console.error('Error syncing live portfolio:', error);
            throw error;
        }
    },

    // ===== Vetting predictions (Market Scan backend) =====
    async getPredictions(symbols: string[] = ['RELIANCE.NS'], horizon: string = 'intraday'): Promise<any> {
        try {
            const syms = symbols.length ? symbols.join(',') : 'RELIANCE.NS';
            const response = await api.get(`/predictions?symbols=${encodeURIComponent(syms)}&horizon=${encodeURIComponent(horizon)}`);
            return response.data;
        } catch (error) {
            console.error('Error fetching predictions:', error);
            throw error;
        }
    },

    // ===== Place order (buy/sell) =====
    async placeOrder(symbol: string, side: 'BUY' | 'SELL', quantity: number, orderType: string = 'MARKET', price?: number): Promise<any> {
        try {
            const response = await api.post('/order', { symbol, side, quantity, order_type: orderType, price });
            return response.data;
        } catch (error) {
            console.error('Error placing order:', error);
            throw error;
        }
    },

    // ===== Health Check =====
    async healthCheck(): Promise<boolean> {
        try {
            const response = await api.get('/status');
            return response.status === 200;
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    },

    // ===== Production-level API calls =====
    async getSignalPerformance(): Promise<any> {
        try {
            const response = await api.get('/production/signal-performance');
            return response.data;
        } catch (error) {
            console.error('Error getting signal performance:', error);
            throw error;
        }
    },

    async getRiskMetrics(): Promise<any> {
        try {
            const response = await api.get('/production/risk-metrics');
            return response.data;
        } catch (error) {
            console.error('Error getting risk metrics:', error);
            throw error;
        }
    },

    async makeProductionDecision(symbol: string): Promise<any> {
        try {
            const response = await api.post('/production/make-decision', { symbol });
            return response.data;
        } catch (error) {
            console.error('Error making production decision:', error);
            throw error;
        }
    },

    async getLearningInsights(): Promise<any> {
        try {
            const response = await api.get('/production/learning-insights');
            return response.data;
        } catch (error) {
            console.error('Error getting learning insights:', error);
            throw error;
        }
    },

    async getDecisionHistory(days: number = 7): Promise<any> {
        try {
            const response = await api.get(`/production/decision-history?days=${days}`);
            return response.data;
        } catch (error) {
            console.error('Error getting decision history:', error);
            throw error;
        }
    },
};

// ===== SSE Stream =====
export function createBotStream(
    onLog: (level: string, message: string) => void,
    onData: (payload: any) => void,
    onConnected?: () => void,
): () => void {
    const url = `${API_BASE_URL}/api/stream`;
    const es = new EventSource(url);

    es.onopen = () => onConnected?.();

    es.onmessage = (ev) => {
        try {
            const parsed = JSON.parse(ev.data);
            if (parsed.type === 'log') {
                onLog(parsed.level ?? 'INFO', parsed.message ?? '');
            } else if (parsed.type === 'data') {
                onData(parsed.payload);
            } else if (parsed.type === 'connected') {
                onConnected?.();
            }
        } catch {
            // ignore malformed events
        }
    };

    es.onerror = () => {
        // EventSource will auto-reconnect; no action needed
    };

    return () => es.close();
}

// ===== Utility Functions =====
export const formatCurrency = (amount: number | null | undefined): string => {
    // Handle NaN, undefined, null values
    if (amount === null || amount === undefined || isNaN(amount)) {
        return '₹0.00';
    }

    // For sidebar metrics, use compact notation for large numbers
    if (Math.abs(amount) >= 10000000) { // 1 crore
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            notation: 'compact',
            maximumFractionDigits: 2,
        }).format(amount);
    } else if (Math.abs(amount) >= 100000) { // 1 lakh
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            maximumFractionDigits: 0,
        }).format(amount);
    } else {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 2,
        }).format(amount);
    }
};

export const formatPercentage = (value: number | null | undefined): string => {
    // Handle NaN, undefined, null values
    if (value === null || value === undefined || isNaN(value)) {
        return '0.00%';
    }

    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
};

export const formatNumber = (value: number): string => {
    return new Intl.NumberFormat('en-IN').format(value);
};

export default hftApiService;
