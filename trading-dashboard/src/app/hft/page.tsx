'use client';

import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import toast, { Toaster } from 'react-hot-toast';
import HftSidebar from '@/components/hft/HftSidebar';
import HftHeader from '@/components/hft/HftHeader';
import HftDashboard from '@/components/hft/HftDashboard';
import HftPortfolio from '@/components/hft/HftPortfolio';
import HftChatAssistant from '@/components/hft/HftChatAssistant';
import HftLoadingOverlay from '@/components/hft/HftLoadingOverlay';
import HftSettingsModal from '@/components/hft/HftSettingsModal';
import { hftApiService } from '@/services/hftApiService';
import type { HftBotData, HftChatMessage } from '@/types/hft';

const AppContainer = styled.div`
  display: flex;
  min-height: 100vh;
  width: 100%;
  background: #ffffff;
`;

const MainContent = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  overflow-x: hidden;
  display: flex;
  flex-direction: column;
  min-height: 100vh;

  &::-webkit-scrollbar { width: 8px; }
  &::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.05); border-radius: 4px; }
  &::-webkit-scrollbar-thumb { background: rgba(0, 0, 0, 0.2); border-radius: 4px; }
  &::-webkit-scrollbar-thumb:hover { background: rgba(0, 0, 0, 0.3); }
  scrollbar-width: thin;
  scrollbar-color: rgba(0, 0, 0, 0.2) rgba(0, 0, 0, 0.05);
`;

const TabContent = styled.div`
  background: white;
  border-radius: 12px;
  padding: 25px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  flex: 1;
  overflow: visible;
  display: flex;
  flex-direction: column;
  margin-bottom: 20px;
`;

export default function HftPage() {
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

    useEffect(() => {
        initializeApp();
        const interval = setInterval(refreshData, 30000);
        return () => clearInterval(interval);
    }, []);

    const initializeApp = async () => {
        try {
            setLoading(true);
            await loadDataFromBackend();
            await loadLiveStatus();

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
        } finally {
            setLoading(false);
        }
    };

    const loadDataFromBackend = async () => {
        try {
            const data = await hftApiService.getBotData();
            setBotData(prev => ({
                ...prev,
                ...data,
                chatMessages: prev.chatMessages
            }));
        } catch (error) {
            console.error('Error loading data from backend:', error);
            toast.error('Failed to load bot data');
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
        } catch (error) {
            console.error('Error refreshing data:', error);
        }
    };

    const handleStartBot = async () => {
        try {
            setLoading(true);
            await hftApiService.startBot();
            toast.success('Bot started successfully!');
            await refreshData();
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
            await hftApiService.addToWatchlist(ticker);
            toast.success(`Added ${ticker} to watchlist`);
            await refreshData();
        } catch (error) {
            console.error('Error adding ticker:', error);
            toast.error('Failed to add ticker');
        }
    };

    const handleRemoveTicker = async (ticker: string) => {
        try {
            await hftApiService.removeFromWatchlist(ticker);
            toast.success(`Removed ${ticker} from watchlist`);
            await refreshData();
        } catch (error) {
            console.error('Error removing ticker:', error);
            toast.error('Failed to remove ticker');
        }
    };

    const handleSaveSettings = async (settings: any) => {
        try {
            await hftApiService.updateSettings(settings);
            toast.success('Settings saved successfully!');
            setShowSettings(false);
            await refreshData();
        } catch (error) {
            console.error('Error saving settings:', error);
            toast.error('Failed to save settings');
        }
    };

    return (
        <>
            <Toaster position="top-right" />
            <AppContainer>
                <HftSidebar
                    botData={botData}
                    onStartBot={handleStartBot}
                    onStopBot={handleStopBot}
                    onRefresh={refreshData}
                />

                <MainContent>
                    <HftHeader
                        activeTab={activeTab}
                        onTabChange={setActiveTab}
                        botData={botData}
                        liveStatus={liveStatus}
                        onOpenSettings={() => setShowSettings(true)}
                    />

                    <TabContent>
                        {activeTab === 'dashboard' && <HftDashboard botData={botData} />}
                        {activeTab === 'portfolio' && (
                            <HftPortfolio
                                botData={botData}
                                onAddTicker={handleAddTicker}
                                onRemoveTicker={handleRemoveTicker}
                            />
                        )}
                        {activeTab === 'chat' && (
                            <HftChatAssistant
                                messages={botData.chatMessages}
                                onSendMessage={handleSendMessage}
                            />
                        )}
                    </TabContent>
                </MainContent>

                {loading && <HftLoadingOverlay />}
                {showSettings && (
                    <HftSettingsModal
                        settings={botData.config}
                        onSave={handleSaveSettings}
                        onClose={() => setShowSettings(false)}
                    />
                )}
            </AppContainer>
        </>
    );
}
