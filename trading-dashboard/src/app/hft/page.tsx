'use client';

import React, { useState, useEffect } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import Layout from '@/components/Layout';
import { useTheme } from '@/contexts/ThemeContext';
import HftDashboard from '@/components/hft/HftDashboard';
import HftPortfolio from '@/components/hft/HftPortfolio';
import HftChatAssistant from '@/components/hft/HftChatAssistant';
import HftLoadingOverlay from '@/components/hft/HftLoadingOverlay';
import HftSettingsModal from '@/components/hft/HftSettingsModal';
import { hftApiService, formatCurrency, formatPercentage, createBotStream } from '@/services/hftApiService';
import { userAPI } from '@/services/api';
import type { HftBotData, HftChatMessage } from '@/types/hft';
import { CheckCircle2, AlertCircle, RefreshCw, Play, Square, LayoutDashboard, Briefcase, MessageCircle } from 'lucide-react';

export default function HftPage() {
    const { theme } = useTheme();
    const isLight = theme === 'light';
    const isSpace = theme === 'space';

    const [activeTab, setActiveTab] = useState<'dashboard' | 'portfolio' | 'chat'>('dashboard');
    const [botData, setBotData] = useState<HftBotData>({
        portfolio: {
            totalValue: 1000000,
            cash: 1000000,
            holdings: {},
            tradeLog: [],
            startingBalance: 1000000
        },
        config: {
            mode: 'paper',
            tickers: [],
            riskLevel: 'MEDIUM',
            maxAllocation: 0.25
        },
        isRunning: false,
        chatMessages: []
    });

    const [loading, setLoading] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const [liveStatus, setLiveStatus] = useState<any>(null);
    const [connected, setConnected] = useState(false);

    // SSE stream: connect once and keep alive; also do an initial REST load
    useEffect(() => {
        initializeApp();

        const stopStream = createBotStream(
            (_level, _message) => { /* logs no longer shown in UI */ },
            (payload) => {
                // Live bot data snapshot from SSE — never overwrite non-zero cached values with 0
                setConnected(true);
                setBotData(prev => {
                    const prevHoldings = prev.portfolio.holdings || {};
                    const newHoldings = payload.holdings && Object.keys(payload.holdings).length > 0
                        ? payload.holdings
                        : prevHoldings;

                    // Compute totalValue from holdings + cash as a safety net when backend sends 0
                    const rawCash = (payload.cash != null && payload.cash > 0) ? payload.cash : prev.portfolio.cash;
                    const rawTotal = (payload.totalValue != null && payload.totalValue > 0) ? payload.totalValue : prev.portfolio.totalValue;
                    // If still 0 but holdings exist, derive it
                    const holdingsMarketValue = Object.values(newHoldings).reduce((sum: number, h: any) => {
                        const price = h.currentPrice || h.avgPrice || 0;
                        const qty = h.quantity || h.qty || 0;
                        return sum + price * qty;
                    }, 0);
                    const derivedTotal = rawTotal > 0 ? rawTotal : (rawCash + holdingsMarketValue) || prev.portfolio.totalValue;

                    return {
                        ...prev,
                        isRunning: payload.isRunning ?? prev.isRunning,
                        portfolio: {
                            ...prev.portfolio,
                            cash: rawCash,
                            totalValue: derivedTotal,
                            unrealizedPnL: payload.unrealizedPnL ?? prev.portfolio.unrealizedPnL,
                            realizedPnL: payload.realizedPnL ?? prev.portfolio.realizedPnL,
                            holdings: newHoldings,
                        },
                        analysis: payload.analysis ?? prev.analysis,
                    };
                });
            },
            () => setConnected(true),
        );

        // Fallback polling every 60 s (much less aggressive now that SSE handles live updates)
        const interval = setInterval(refreshData, 60000);
        return () => {
            stopStream();
            clearInterval(interval);
        };
    }, []);

    const initializeApp = async () => {
        try {
            setLoading(true);
            await loadDataFromBackend();
            await loadLiveStatus();
            setConnected(true);
            if (botData.chatMessages.length === 0) {
                setBotData(prev => ({
                    ...prev,
                    chatMessages: [{
                        role: 'assistant',
                        content: 'Welcome to the Indian Stock Trading Bot! 🚀\nType a command or ask me anything about trading and markets.',
                        timestamp: new Date().toISOString()
                    }]
                }));
            }
        } catch (error) {
            console.error('Error initializing app:', error);
            toast.error('Failed to initialize application');
            setConnected(false);
        } finally {
            setLoading(false);
        }
    };

    const loadDataFromBackend = async () => {
        try {
            const data = await hftApiService.getBotData();
            // Ensure mode is properly set from backend response
            const backendMode = data?.config?.mode || 'paper';
            // Also fetch watchlist directly to ensure we have the latest
            let watchlistTickers: string[] = [];
            try {
                watchlistTickers = await hftApiService.getWatchlist();
            } catch (watchlistErr) {
                // Fallback to tickers from bot data if watchlist endpoint fails
                watchlistTickers = data?.config?.tickers || [];
            }
            setBotData(prev => ({
                ...prev,
                ...data,
                config: {
                    ...prev.config,
                    ...data.config,
                    mode: backendMode,  // Use mode from backend
                    tickers: watchlistTickers  // Use watchlist from dedicated endpoint
                },
                chatMessages: prev.chatMessages
            }));
            setConnected(true);
            // Update live status based on mode
            if (backendMode === 'live') {
                await loadLiveStatus();
            }
        } catch (error: any) {
            // A timeout means the backend is slow/busy, NOT offline — keep previous data visible
            const isTimeout = error?.message?.includes('timeout') || error?.code === 'ECONNABORTED';
            const isNetworkError = error?.message === 'Network Error' || error?.code === 'ERR_NETWORK';
            if (isNetworkError) {
                // Only mark offline for true connection failures
                setConnected(false);
            }
            // For timeouts or other transient errors, silently keep last known state
            if (!isTimeout && !isNetworkError) {
                console.warn('Non-timeout bot data error, trying settings fallback:', error?.message);
                setConnected(false);
                try {
                    const settings = await hftApiService.getSettings();
                    const savedMode = settings?.mode || 'paper';
                    setBotData(prev => ({
                        ...prev,
                        config: { ...prev.config, mode: savedMode }
                    }));
                } catch { /* keep last state */ }
            }
        }
    };

    const loadLiveStatus = async () => {
        try {
            const status = await hftApiService.getLiveStatus();
            setLiveStatus(status);
        } catch (error) {
            console.error('Error loading live status:', error);
        }
    };

    const refreshData = async () => {
        try {
            await loadDataFromBackend();
            await loadLiveStatus();
            try {
                await hftApiService.syncLivePortfolio();
            } catch { /* optional */ }
        } catch (error) {
            console.error('Error refreshing data:', error);
        }
    };

    const handlePlaceOrder = async (symbol: string, side: 'BUY' | 'SELL', quantity: number) => {
        try {
            const res = await hftApiService.placeOrder(symbol, side, quantity, 'MARKET');
            const msg = res?.message || res?.detail?.message || `${side} order placed for ${symbol}`;
            toast.success(msg);
            await refreshData();
        } catch (error: any) {
            console.error('Place order error:', error);
            toast.error(error?.message || error?.response?.data?.detail || 'Order failed');
        }
    };

    const handleStartBot = async () => {
        try {
            setLoading(true);
            // 1. Load this user's personal watchlist from MongoDB
            const userTickers = await userAPI.getWatchlist();
            // 2. If the user has tickers, sync them to the bot before starting
            if (userTickers.length > 0) {
                try {
                    await hftApiService.bulkUpdateWatchlist(userTickers, 'ADD');
                } catch {
                    // If bulk update fails, still try to start
                }
            }
            // 3. Start the bot (it now has the user's tickers)
            await hftApiService.startBot();
            // 4. Optimistically mark as running immediately — backend starts in background
            // so refreshData won't yet show isRunning:true; we set it here so Stop Bot is clickable
            setBotData(prev => ({ ...prev, isRunning: true }));
            toast.success('Bot started! Analysis running in background...');
            // 5. Refresh after a short delay to pick up backend state
            setTimeout(() => refreshData(), 3000);
        } catch (error) {
            console.error('Error starting bot:', error);
            toast.error('Failed to start bot');
        } finally {
            setLoading(false);
        }
    };

    const handleStopBot = async () => {
        try {
            setLoading(true);
            await hftApiService.stopBot();
            // Immediately mark as stopped
            setBotData(prev => ({ ...prev, isRunning: false }));
            toast.success('Bot stopped successfully!');
            await refreshData();
        } catch (error) {
            console.error('Error stopping bot:', error);
            toast.error('Failed to stop bot');
        } finally {
            setLoading(false);
        }
    };

    const handleSendMessage = async (message: string) => {
        try {
            const userMessage: HftChatMessage = {
                role: 'user',
                content: message,
                timestamp: new Date().toISOString()
            };
            setBotData(prev => ({
                ...prev,
                chatMessages: [...prev.chatMessages, userMessage]
            }));
            const response = await hftApiService.sendChatMessage(message);
            const assistantMessage: HftChatMessage = {
                role: 'assistant',
                content: response.response,
                timestamp: new Date().toISOString()
            };
            setBotData(prev => ({
                ...prev,
                chatMessages: [...prev.chatMessages, assistantMessage]
            }));
        } catch (error) {
            console.error('Error sending message:', error);
            toast.error('Failed to send message');
        }
    };

    const handleAddTicker = async (ticker: string) => {
        try {
            // Normalize ticker format
            const normalizedTicker = ticker.toUpperCase().trim();
            const tickerToAdd = normalizedTicker.endsWith('.NS') || normalizedTicker.endsWith('.BO')
                ? normalizedTicker
                : normalizedTicker + '.NS';

            // Call backend API
            const response = await hftApiService.addToWatchlist(tickerToAdd);

            // Update UI immediately with response data
            setBotData(prev => ({
                ...prev,
                config: {
                    ...prev.config,
                    tickers: response.tickers || []
                }
            }));

            toast.success(response.message || `Added ${tickerToAdd} to watchlist`);
        } catch (error) {
            console.error('Error adding ticker:', error);
            toast.error('Failed to add ticker');
            // Refresh to get correct state on error
            await refreshData();
        }
    };

    const handleRemoveTicker = async (ticker: string) => {
        try {
            // Normalize ticker format
            const normalizedTicker = ticker.toUpperCase().trim();
            const tickerToRemove = normalizedTicker.endsWith('.NS') || normalizedTicker.endsWith('.BO')
                ? normalizedTicker
                : normalizedTicker + '.NS';

            // Call backend API
            const response = await hftApiService.removeFromWatchlist(tickerToRemove);

            // Update UI immediately with response data
            setBotData(prev => ({
                ...prev,
                config: {
                    ...prev.config,
                    tickers: response.tickers || []
                }
            }));

            toast.success(response.message || `Removed ${tickerToRemove} from watchlist`);
        } catch (error) {
            console.error('Error removing ticker:', error);
            toast.error('Failed to remove ticker');
            // Refresh to get correct state on error
            await refreshData();
        }
    };

    const handleSaveSettings = async (settings: any) => {
        try {
            setLoading(true);
            await hftApiService.updateSettings(settings);
            toast.success('Settings saved successfully!');
            setShowSettings(false);
            // Refresh data multiple times to ensure live mode is reflected
            await refreshData();
            await new Promise(resolve => setTimeout(resolve, 500)); // Wait 500ms
            await refreshData();
            await loadLiveStatus(); // Explicitly reload live status
        } catch (error) {
            console.error('Error saving settings:', error);
            toast.error('Failed to save settings');
        } finally {
            setLoading(false);
        }
    };

    const mode = (liveStatus?.mode ?? botData?.config?.mode) || 'paper';
    const cash = botData.portfolio.cash || 0;
    // If backend sent totalValue=0 but holdings exist, derive it from holdings market value + cash
    const holdingsMarketValuePage = Object.values(botData.portfolio.holdings || {}).reduce((sum, h: any) => {
        const price = h.currentPrice || h.avgPrice || 0;
        const qty = h.quantity || h.qty || 0;
        return sum + price * qty;
    }, 0);
    const totalValue = (botData.portfolio.totalValue || 0) > 0
        ? botData.portfolio.totalValue
        : (cash + holdingsMarketValuePage) || 0;
    const startingBalance = botData.portfolio.startingBalance || totalValue;
    const cashInvested = startingBalance - cash;
    const totalReturn = totalValue - startingBalance;
    const returnPercentage = startingBalance > 0 ? (totalReturn / startingBalance) * 100 : 0;
    const positionsCount = Object.keys(botData.portfolio.holdings).length;

    const cardBg = isLight ? 'bg-white' : isSpace ? 'bg-slate-800/80' : 'bg-slate-800';
    const cardBorder = isLight ? 'border-gray-200' : isSpace ? 'border-purple-900/30' : 'border-slate-700';
    const textPrimary = isLight ? 'text-gray-900' : 'text-white';
    const textMuted = isLight ? 'text-gray-600' : 'text-gray-400';

    return (
        <>
            <Toaster position="top-right" />
            <Layout>
                <div className={`space-y-3 md:space-y-4 w-full ${isLight ? '' : 'animate-fadeIn'}`}>
                    {/* Header: title + status + refresh (same structure as main dashboard) */}
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                                <h1 className={`text-xl md:text-2xl font-bold ${textPrimary}`}>BOT</h1>
                                <div className="flex items-center gap-2">
                                    {connected ? (
                                        <div className="flex items-center gap-1 px-2 py-0.5 bg-green-500/20 border border-green-500/50 rounded-lg">
                                            <CheckCircle2 className="w-3 h-3 text-green-400" />
                                            <span className="text-green-400 text-xs font-medium">Connected</span>
                                        </div>
                                    ) : (
                                        <div className="flex items-center gap-1 px-2 py-0.5 bg-red-500/20 border border-red-500/50 rounded-lg">
                                            <AlertCircle className="w-3 h-3 text-red-400" />
                                            <span className="text-red-400 text-xs font-medium">Offline</span>
                                        </div>
                                    )}
                                    <div className={`flex items-center gap-1 px-2 py-0.5 rounded-lg border ${mode === 'live' ? 'bg-red-500/20 border-red-500/50' : 'bg-blue-500/20 border-blue-500/50'}`}>
                                        <span className={`w-2 h-2 rounded-full ${mode === 'live' ? 'bg-red-400' : 'bg-blue-400'}`} />
                                        <span className={`${mode === 'live' ? 'text-red-400' : 'text-blue-400'} text-xs font-medium`}>
                                            {mode === 'live' ? 'Live Trading' : 'Paper Trading'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <p className={`text-xs md:text-sm ${textMuted}`}>
                                Updated {new Date().toLocaleTimeString()}
                            </p>
                            {mode === 'live' && (botData as any).dhan_error && (
                                <p className="text-xs mt-1 text-amber-500 dark:text-amber-400">
                                    Dhan fetch failed: {(botData as any).dhan_error}
                                </p>
                            )}
                        </div>
                        <button
                            onClick={refreshData}
                            disabled={loading}
                            className="flex items-center justify-center gap-1.5 px-4 py-2.5 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm font-semibold transition-all disabled:opacity-50 w-full md:w-auto min-h-[44px] md:min-h-0"
                        >
                            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                            Refresh
                        </button>
                    </div>

                    {/* Portfolio metrics row (same idea as main dashboard cards) */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className={`${cardBg} border ${cardBorder} rounded-xl p-4`}>
                            <p className={`text-xs font-medium uppercase tracking-wide ${textMuted}`}>Total Value</p>
                            <p className={`text-lg font-bold ${textPrimary}`}>{formatCurrency(totalValue)}</p>
                        </div>
                        <div className={`${cardBg} border ${cardBorder} rounded-xl p-4`}>
                            <p className={`text-xs font-medium uppercase tracking-wide ${textMuted}`}>Cash</p>
                            <p className={`text-lg font-bold ${textPrimary}`}>{formatCurrency(cash)}</p>
                        </div>
                        <div className={`${cardBg} border ${cardBorder} rounded-xl p-4`}>
                            <p className={`text-xs font-medium uppercase tracking-wide ${textMuted}`}>Total Return</p>
                            <p className={`text-lg font-bold ${textPrimary}`}>{formatCurrency(totalReturn)}</p>
                            <p className={`text-sm font-semibold ${returnPercentage >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                                {formatPercentage(returnPercentage)}
                            </p>
                        </div>
                        <div className={`${cardBg} border ${cardBorder} rounded-xl p-4`}>
                            <p className={`text-xs font-medium uppercase tracking-wide ${textMuted}`}>Positions</p>
                            <p className={`text-lg font-bold ${textPrimary}`}>{positionsCount}</p>
                        </div>
                    </div>

                    {/* Quick actions */}
                    <div className="flex flex-wrap gap-2">
                        <button
                            onClick={handleStartBot}
                            disabled={botData.isRunning}
                            className="flex items-center gap-2 px-4 py-2.5 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-semibold transition-all"
                        >
                            <Play className="w-4 h-4" /> Start Bot
                        </button>
                        <button
                            onClick={handleStopBot}
                            disabled={!botData.isRunning}
                            className="flex items-center gap-2 px-4 py-2.5 bg-red-600 hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-sm font-semibold transition-all"
                        >
                            <Square className="w-4 h-4" /> Stop Bot
                        </button>
                        <button
                            onClick={() => setShowSettings(true)}
                            disabled={botData.isRunning}
                            className="flex items-center gap-2 px-4 py-2.5 bg-slate-600 hover:bg-slate-700 disabled:opacity-50 text-white rounded-lg text-sm font-semibold transition-all"
                        >
                            Settings
                        </button>
                    </div>

                    {/* Tabs (Dashboard / Portfolio / Chat) */}
                    <div className={`${cardBg} border ${cardBorder} rounded-xl overflow-hidden`}>
                        <div className={`flex border-b ${cardBorder} p-1 gap-1`}>
                            {[
                                { id: 'dashboard' as const, label: 'Dashboard', icon: LayoutDashboard },
                                { id: 'portfolio' as const, label: 'Portfolio', icon: Briefcase },
                                { id: 'chat' as const, label: 'Chat Assistant', icon: MessageCircle }
                            ].map(({ id, label, icon: Icon }) => (
                                <button
                                    key={id}
                                    onClick={() => setActiveTab(id)}
                                    className={`flex items-center gap-2 px-4 py-3 rounded-lg text-sm font-medium transition-all ${activeTab === id
                                        ? isLight ? 'bg-blue-500 text-white' : 'bg-blue-600 text-white'
                                        : isLight ? 'text-gray-600 hover:bg-gray-100' : 'text-gray-400 hover:bg-slate-700'
                                        }`}
                                >
                                    <Icon className="w-4 h-4" /> {label}
                                </button>
                            ))}
                        </div>
                        <div className="p-4 md:p-6 min-h-[400px]">
                            {/* Always show components regardless of trading mode or connection status */}
                            {activeTab === 'dashboard' && <HftDashboard botData={botData} onPlaceOrder={handlePlaceOrder} onRefresh={refreshData} />}
                            {activeTab === 'portfolio' && (
                                <HftPortfolio
                                    botData={botData}
                                    onAddTicker={handleAddTicker}
                                    onRemoveTicker={handleRemoveTicker}
                                    onRefresh={refreshData}
                                />
                            )}
                            {activeTab === 'chat' && (
                                <HftChatAssistant
                                    messages={botData.chatMessages}
                                    onSendMessage={handleSendMessage}
                                />
                            )}
                        </div>
                    </div>
                </div>

                {loading && <HftLoadingOverlay />}
                {showSettings && (
                    <HftSettingsModal
                        settings={botData.config}
                        onSave={handleSaveSettings}
                        onClose={() => setShowSettings(false)}
                    />
                )}
            </Layout>
        </>
    );
}
