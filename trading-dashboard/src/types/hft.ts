// HFT API Service Types
export interface HftBotConfig {
    mode: 'paper' | 'live';
    tickers: string[];
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
    maxAllocation: number;
}

export interface HftPortfolio {
    totalValue: number;
    cash: number;
    holdings: Record<string, HftHolding>;
    tradeLog: HftTrade[];
    startingBalance: number;
}

export interface HftHolding {
    symbol: string;
    quantity: number;
    avgPrice: number;
    currentPrice?: number;
    value?: number;
    pnl?: number;
    pnlPercent?: number;
}

export interface HftTrade {
    symbol: string;
    action: 'BUY' | 'SELL';
    quantity: number;
    price: number;
    timestamp: string;
    total: number;
    portfolioValue?: number;
}

export interface HftBotData {
    portfolio: HftPortfolio;
    config: HftBotConfig;
    isRunning: boolean;
    chatMessages: HftChatMessage[];
    /** Set when Live mode + Dhan configured but portfolio fetch failed (e.g. token/network). */
    dhan_error?: string;
}

export interface HftChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
}

export interface HftLiveStatus {
    connected: boolean;
    mode?: string;
    lastUpdate?: string;
    dhan_configured?: boolean;
    /** Error from last Dhan portfolio fetch when in Live mode. */
    dhan_error?: string | null;
    broker?: string;
    account?: string;
    lastSync?: string;
    error?: string;
}

export interface HftMcpStatus {
    mcp_available: boolean;
    server_initialized: boolean;
}

export interface HftMcpAnalysisRequest {
    symbol: string;
    timeframe?: string;
    analysis_type?: string;
}

export interface HftMcpAnalysisResponse {
    recommendation: string;
    confidence: number;
    current_price: number;
    target_price: number;
    stop_loss: number;
    reasoning?: string;
}

export interface HftMcpChatRequest {
    message: string;
    context?: {
        type?: string;
        [key: string]: any;
    };
}

export interface HftMcpChatResponse {
    response: string;
    timestamp: string;
}

export interface HftWatchlistResponse {
    message: string;
    tickers: string[];
}

export interface HftSettingsUpdate {
    mode?: 'paper' | 'live';
    riskLevel?: 'LOW' | 'MEDIUM' | 'HIGH';
    maxAllocation?: number;
    stopLoss?: number;
}
