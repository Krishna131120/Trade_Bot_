import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import toast from 'react-hot-toast';
import type { HftBotData, HftTrade } from '../../types/hft';
import { formatCurrency, hftApiService } from '../../services/hftApiService';

const PortfolioContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
`;

const SubTabNav = styled.div`
  display: flex;
  background: white;
  border-radius: 10px;
  padding: 5px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
`;

const SubTabButton = styled.button<{ $active: boolean }>`
  flex: 1;
  background: ${props => props.$active ? '#3498db' : 'transparent'};
  border: none;
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.2s ease;
  color: ${props => props.$active ? 'white' : '#7f8c8d'};
  box-shadow: ${props => props.$active ? '0 2px 8px rgba(52, 152, 219, 0.25)' : 'none'};

  &:hover {
    background: ${props => props.$active ? '#2980b9' : '#ecf0f1'};
    color: #2c3e50;
  }
`;

const TabBody = styled.div`
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const TabPanel = styled.div`
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 6px;
`;

const Section = styled.div`
  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 5px;
  }
`;

const HoldingsTable = styled.div`
  background: #f8f9fa;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #e9ecef;
  max-height: 80vh;
  overflow-y: auto;

  &::-webkit-scrollbar { width: 8px; }
  &::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }
  &::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
  &::-webkit-scrollbar-thumb:hover { background: #a8a8a8; }
`;

const HoldingsHeader = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 0.8fr 1fr 1fr 1fr 1fr 1fr 0.8fr 1.2fr;
  gap: 10px;
  padding: 15px;
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  font-weight: bold;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr 0.8fr 1fr 1fr;
    font-size: 0.7rem;
    gap: 5px;
  }
`;

const HoldingsRow = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 0.8fr 1fr 1fr 1fr 1fr 1fr 0.8fr 1.2fr;
  gap: 10px;
  padding: 15px;
  align-items: center;
  border-bottom: 1px solid #e9ecef;
  background: white;
  transition: all 0.2s ease;

  &:last-child { border-bottom: none; }
  &:hover { background: #f8f9fa; transform: translateY(-1px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }

  @media (max-width: 768px) {
    grid-template-columns: 1fr 0.8fr 1fr 1fr;
    font-size: 0.8rem;
    gap: 5px;
  }
`;

const TickerName = styled.div`
  font-weight: bold;
`;

const ProfitLoss = styled.div<{ value: number }>`
  font-weight: bold;
  color: ${props => props.value > 0 ? '#27ae60' : props.value < 0 ? '#e74c3c' : '#7f8c8d'};
`;

const TradeBadge = styled.span<{ type: string }>`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: ${props => props.type === 'BUY' ? '#27ae60' : '#e74c3c'};
  color: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
`;

const StatusBadge = styled.span<{ status: string }>`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  background: ${props => {
        switch (props.status) {
            case 'ACTIVE': return '#27ae60';
            case 'PROFIT': return '#2ecc71';
            case 'LOSS': return '#e74c3c';
            default: return '#3498db';
        }
    }};
`;

const LastUpdateTime = styled.div`
  font-size: 0.8rem;
  color: #7f8c8d;
  text-align: right;
  margin-bottom: 10px;
  font-style: italic;
`;

const RealTimeIndicator = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: #27ae60;
  margin-bottom: 10px;

  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #27ae60;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

const WatchlistControls = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  @media(max-width: 768px) { flex-direction: column; }
`;

const TickerInput = styled.input`
  flex: 1;
  padding: 10px;
  border: 2px solid #e9ecef;
  border-radius: 6px;
  font-size: 1rem;
  &:focus { outline: none; border-color: #3498db; }
`;

const AddButton = styled.button`
  background: #27ae60;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
  &:hover { background: #229954; }
  &:disabled { background: #95a5a6; cursor: not-allowed; }
`;

const WatchlistGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;
  @media(max-width: 768px) { grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); }
`;

const WatchlistItem = styled.div`
  background: #3498db;
  color: white;
  padding: 10px;
  border-radius: 6px;
  text-align: center;
  font-weight: 500;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const RemoveButton = styled.button`
  position: absolute;
  top: -5px;
  right: -5px;
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  cursor: pointer;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  &:hover { background: #c0392b; transform: scale(1.1); }
`;

const NoHoldings = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 20px;
`;

const ActivitySection = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;
  display: flex;
  flex-direction: column;
  min-height: 200px;
`;

const ActivityList = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 80vh;
  min-height: 120px;
  padding-right: 8px;
  margin-right: -8px;
  &::-webkit-scrollbar { width: 8px; }
  &::-webkit-scrollbar-track { background: #f8f9fa; border-radius: 4px; border: 1px solid #e9ecef; }
  &::-webkit-scrollbar-thumb { background: #6c757d; border-radius: 4px; border: 1px solid #dee2e6; }
  &::-webkit-scrollbar-thumb:hover { background: #495057; }
  scrollbar-width: thin;
  scrollbar-color: #6c757d #f8f9fa;
`;

const ActivityItem = styled.div<{ type: string }>`
  background: white;
  padding: 16px;
  margin-bottom: 10px;
  border-radius: 8px;
  border-left: 4px solid ${props => props.type === 'BUY' ? '#27ae60' : '#e74c3c'};
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  min-height: 70px;
`;

const ActivityDetails = styled.div`
  flex: 1;
`;

const ActivityTitle = styled.div<{ type: string }>`
  font-weight: bold;
  margin-bottom: 5px;
  color: ${props => props.type === 'BUY' ? '#27ae60' : '#e74c3c'};
  font-size: 0.95rem;
`;

const ActivityTime = styled.div`
  font-size: 0.9rem;
  color: #666;
`;

const ActivityValues = styled.div`
  text-align: right;
  font-size: 0.85rem;
  min-width: 120px;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
`;

const OrderBtn = styled.button<{ $side: 'BUY' | 'SELL' }>`
  padding: 6px 10px;
  border: none;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 600;
  cursor: pointer;
  background: ${p => p.$side === 'BUY' ? '#27ae60' : '#e74c3c'};
  color: white;
  &:hover { opacity: 0.9; }
  &:disabled { opacity: 0.5; cursor: not-allowed; }
`;

const OrderModalOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const OrderModalBox = styled.div`
  background: white;
  border-radius: 12px;
  padding: 24px;
  min-width: 320px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.2);
`;

const OrderModalTitle = styled.h3`
  margin: 0 0 16px 0;
  color: #2c3e50;
`;

const OrderField = styled.div`
  margin-bottom: 14px;
  label { display: block; margin-bottom: 4px; font-weight: 600; color: #555; font-size: 0.9rem; }
  input { width: 100%; padding: 8px 10px; border: 2px solid #e9ecef; border-radius: 6px; font-size: 1rem; box-sizing: border-box; }
  input:focus { outline: none; border-color: #3498db; }
`;

const OrderModalActions = styled.div`
  display: flex;
  gap: 10px;
  margin-top: 20px;
  button { padding: 10px 18px; border-radius: 6px; font-weight: 600; cursor: pointer; border: none; }
  .cancel { background: #95a5a6; color: white; }
  .submit { background: #3498db; color: white; }
  .submit:hover { background: #2980b9; }
`;

interface HftPortfolioProps {
    botData: HftBotData;
    onAddTicker: (ticker: string) => Promise<void>;
    onRemoveTicker: (ticker: string) => Promise<void>;
}

type OrderSide = 'BUY' | 'SELL';
const HftPortfolio: React.FC<HftPortfolioProps> = ({ botData, onAddTicker, onRemoveTicker }) => {
    const [subTab, setSubTab] = useState<'holdings' | 'activity' | 'watchlist'>('holdings');
    const [newTicker, setNewTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const [tradeHistory, setTradeHistory] = useState<HftTrade[]>([]);
    const [lastUpdate, setLastUpdate] = useState(new Date());
    const [orderModal, setOrderModal] = useState<{ open: boolean; symbol: string; side: OrderSide; quantity: string; stopLossPct: string }>({
        open: false, symbol: '', side: 'BUY', quantity: '', stopLossPct: ''
    });

    useEffect(() => {
        const fetchTradeHistory = async () => {
            try {
                const trades = await hftApiService.getTrades(50);
                setTradeHistory(trades);
            } catch (error) {
                console.error('Error fetching trade history:', error);
            }
        };
        fetchTradeHistory();
    }, []);

    useEffect(() => {
        const interval = setInterval(async () => {
            setLastUpdate(new Date());
            try {
                const trades = await hftApiService.getTrades(50);
                setTradeHistory(trades);
            } catch (error) {
                console.error('Error updating real-time data:', error);
            }
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    const getLastTradeType = (ticker: string): string => {
        const tickerTrades = tradeHistory.filter(trade => trade.symbol === ticker);
        if (tickerTrades.length === 0) return 'BUY';
        const sortedTrades = tickerTrades.sort((a, b) =>
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
        return sortedTrades[0].action;
    };

    const handleAddTicker = async () => {
        if (!newTicker.trim()) {
            alert('Please enter a ticker symbol');
            return;
        }
        if (botData.config.tickers.includes(newTicker.toUpperCase())) {
            alert('Ticker already in watchlist');
            return;
        }
        setLoading(true);
        try {
            await onAddTicker(newTicker.toUpperCase());
            setNewTicker('');
        } catch (error) {
            console.error('Error adding ticker:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleRemoveTicker = async (ticker: string) => {
        setLoading(true);
        try {
            await onRemoveTicker(ticker);
        } catch (error) {
            console.error('Error removing ticker:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            handleAddTicker();
        }
    };

    const openOrderModal = (symbol: string, side: OrderSide) => {
        setOrderModal({ open: true, symbol, side, quantity: '', stopLossPct: '' });
    };

    const closeOrderModal = () => {
        setOrderModal(prev => ({ ...prev, open: false }));
    };

    const handlePlaceOrder = async () => {
        const qty = parseInt(orderModal.quantity, 10);
        if (!orderModal.symbol || isNaN(qty) || qty < 1) {
            toast.error('Enter a valid quantity');
            return;
        }
        setLoading(true);
        try {
            await hftApiService.placeOrder(orderModal.symbol, orderModal.side, qty, 'MARKET');
            toast.success(`${orderModal.side} order placed for ${orderModal.symbol}`);
            closeOrderModal();
            const trades = await hftApiService.getTrades(50);
            setTradeHistory(trades);
        } catch (err) {
            console.error('Place order error:', err);
            toast.error('Failed to place order');
        } finally {
            setLoading(false);
        }
    };

    const calculatePortfolioPercentage = (currentValue: number, totalValue: number): string => {
        if (!totalValue || totalValue <= 0 || currentValue == null) {
            return '0.0';
        }
        const percentage = ((Math.abs(currentValue) / totalValue) * 100);
        return isNaN(percentage) ? '0.0' : percentage.toFixed(1);
    };

    const holdings = botData.portfolio.holdings || {};
    const totalValue = botData.portfolio.totalValue || 0;

    return (
        <PortfolioContainer>
            <SubTabNav>
                <SubTabButton $active={subTab === 'holdings'} onClick={() => setSubTab('holdings')}>
                    <i className="fas fa-briefcase" style={{ marginRight: 8 }}></i> Current Holdings
                </SubTabButton>
                <SubTabButton $active={subTab === 'activity'} onClick={() => setSubTab('activity')}>
                    <i className="fas fa-history" style={{ marginRight: 8 }}></i> Recent Trading Activity
                </SubTabButton>
                <SubTabButton $active={subTab === 'watchlist'} onClick={() => setSubTab('watchlist')}>
                    <i className="fas fa-list" style={{ marginRight: 8 }}></i> Watchlist
                </SubTabButton>
            </SubTabNav>

            <TabBody>
                {subTab === 'holdings' && (
                    <TabPanel>
                        <Section>
                            <h3>Current Holdings</h3>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                                <RealTimeIndicator>
                                    Live Portfolio Data
                                </RealTimeIndicator>
                                <LastUpdateTime>
                                    Last updated: {lastUpdate.toLocaleTimeString()}
                                </LastUpdateTime>
                            </div>

                            <HoldingsTable>
                                {Object.keys(holdings).length > 0 ? (
                                    <>
                                        <HoldingsHeader>
                                            <div>Stock Symbol</div>
                                            <div>Status</div>
                                            <div>Last Trade</div>
                                            <div>Quantity</div>
                                            <div>Avg Price</div>
                                            <div>Current Price</div>
                                            <div>Profit/Loss</div>
                                            <div>% Portfolio</div>
                                            <div>Actions</div>
                                        </HoldingsHeader>
                                        {Object.entries(holdings).map(([ticker, data]) => {
                                            const currentPrice = data.currentPrice || data.avgPrice || 0;
                                            const avgPrice = data.avgPrice || 0;
                                            const qty = data.quantity || 0;
                                            const currentValue = qty * currentPrice;
                                            const costBasis = qty * avgPrice;
                                            const profitLoss = currentValue - costBasis;
                                            const profitLossPct = costBasis > 0 ? ((profitLoss / costBasis) * 100) : 0;
                                            const portfolioPercentage = calculatePortfolioPercentage(currentValue, totalValue);
                                            const lastTradeType = getLastTradeType(ticker);
                                            const status = qty > 0 ? (profitLoss >= 0 ? 'PROFIT' : 'LOSS') : 'ACTIVE';
                                            return (
                                                <HoldingsRow key={ticker}>
                                                    <TickerName>{ticker}</TickerName>
                                                    <div>
                                                        <StatusBadge status={status}>{status}</StatusBadge>
                                                    </div>
                                                    <div>
                                                        <TradeBadge type={lastTradeType}>{lastTradeType}</TradeBadge>
                                                    </div>
                                                    <div>{qty.toFixed(2)}</div>
                                                    <div>{formatCurrency(avgPrice)}</div>
                                                    <div>{formatCurrency(currentPrice)}</div>
                                                    <ProfitLoss value={profitLoss}>
                                                        {profitLoss >= 0 ? '+' : ''}{formatCurrency(profitLoss)}
                                                        <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
                                                            ({profitLossPct >= 0 ? '+' : ''}{profitLossPct.toFixed(2)}%)
                                                        </div>
                                                    </ProfitLoss>
                                                    <div>{portfolioPercentage}%</div>
                                                    <ActionButtons>
                                                        <OrderBtn $side="BUY" onClick={() => openOrderModal(ticker, 'BUY')} disabled={loading}>Buy</OrderBtn>
                                                        <OrderBtn $side="SELL" onClick={() => openOrderModal(ticker, 'SELL')} disabled={loading}>Sell</OrderBtn>
                                                    </ActionButtons>
                                                </HoldingsRow>
                                            );
                                        })}
                                    </>
                                ) : (
                                    <NoHoldings>No current holdings</NoHoldings>
                                )}
                            </HoldingsTable>
                        </Section>
                    </TabPanel>
                )}

                {subTab === 'activity' && (
                    <TabPanel>
                        <Section>
                            <h3>Recent Trading Activity</h3>
                            <ActivitySection>
                                <ActivityList>
                                    {tradeHistory && tradeHistory.length > 0 ? (
                                        tradeHistory.slice(0, 50).map((trade, index) => {
                                            const action = trade.action.toUpperCase();
                                            const displayAsset = trade.symbol.replace('.NS', '');
                                            const qty = trade.quantity;
                                            const price = trade.price;
                                            const d = new Date(trade.timestamp);
                                            const formatted = `${d.toLocaleDateString('en-IN')} ${d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}`;
                                            return (
                                                <ActivityItem key={`act-${index}-${trade.symbol}-${trade.timestamp}`} type={action}>
                                                    <ActivityDetails>
                                                        <ActivityTitle type={action}>{action} {displayAsset}</ActivityTitle>
                                                        <ActivityTime>{formatted}</ActivityTime>
                                                    </ActivityDetails>
                                                    <ActivityValues>
                                                        <div><strong>Qty:</strong> {qty}</div>
                                                        <div><strong>Price:</strong> {formatCurrency(price)}</div>
                                                        <div><strong>Total:</strong> {formatCurrency(qty * price)}</div>
                                                    </ActivityValues>
                                                </ActivityItem>
                                            );
                                        })
                                    ) : (
                                        <div style={{ textAlign: 'center', color: '#7f8c8d', fontStyle: 'italic', padding: '20px', background: 'white', borderRadius: '8px', border: '2px dashed #e9ecef' }}>
                                            No recent trades
                                        </div>
                                    )}
                                </ActivityList>
                            </ActivitySection>
                        </Section>
                    </TabPanel>
                )}

                {subTab === 'watchlist' && (
                    <TabPanel>
                        <Section>
                            <h3>Watchlist Management</h3>
                            <WatchlistControls>
                                <TickerInput
                                    type="text"
                                    placeholder="Add Ticker (e.g., INFY.NS)"
                                    value={newTicker}
                                    onChange={(e) => setNewTicker(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    disabled={loading}
                                />
                                <AddButton onClick={handleAddTicker} disabled={loading || !newTicker.trim()}>
                                    {loading ? 'Adding...' : 'Add Ticker'}
                                </AddButton>
                            </WatchlistControls>

                            <div>
                                <h4 style={{ color: '#2c3e50', marginBottom: 10 }}>Current Watchlist:</h4>
                                <WatchlistGrid>
                                    {botData.config.tickers.map(ticker => (
                                        <WatchlistItem key={ticker}>
                                            {ticker}
                                            <RemoveButton onClick={() => handleRemoveTicker(ticker)}>×</RemoveButton>
                                        </WatchlistItem>
                                    ))}
                                </WatchlistGrid>
                            </div>
                        </Section>
                    </TabPanel>
                )}
            </TabBody>

            {orderModal.open && (
                <OrderModalOverlay onClick={closeOrderModal}>
                    <OrderModalBox onClick={e => e.stopPropagation()}>
                        <OrderModalTitle>{orderModal.side} {orderModal.symbol}</OrderModalTitle>
                        <OrderField>
                            <label>Quantity</label>
                            <input
                                type="number"
                                min={1}
                                value={orderModal.quantity}
                                onChange={e => setOrderModal(prev => ({ ...prev, quantity: e.target.value }))}
                                placeholder="Number of shares"
                            />
                        </OrderField>
                        <OrderField>
                            <label>Stop Loss % (optional)</label>
                            <input
                                type="number"
                                min={0}
                                step={0.1}
                                value={orderModal.stopLossPct}
                                onChange={e => setOrderModal(prev => ({ ...prev, stopLossPct: e.target.value }))}
                                placeholder="e.g. 2"
                            />
                        </OrderField>
                        <OrderModalActions>
                            <button type="button" className="cancel" onClick={closeOrderModal}>Cancel</button>
                            <button type="button" className="submit" onClick={handlePlaceOrder} disabled={loading}>
                                {loading ? 'Placing...' : 'Place Order'}
                            </button>
                        </OrderModalActions>
                    </OrderModalBox>
                </OrderModalOverlay>
            )}
        </PortfolioContainer>
    );
};

export default HftPortfolio;
