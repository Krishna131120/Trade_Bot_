import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
    Filler
} from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';
import type { HftBotData } from '../../types/hft';
import { formatCurrency, hftApiService } from '../../services/hftApiService';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement, Filler);

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-height: 100%;
  overflow: visible;
  padding-bottom: 20px;
`;

const MetricsRow = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
`;

const MetricCard = styled.div`
  background: white;
  color: #333;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  padding: 20px;
  border-radius: 12px;
  text-align: center;
`;

const MetricLabel = styled.div`
  font-size: 0.9rem;
  color: #7f8c8d;
  margin-bottom: 8px;
`;

const MetricValue = styled.div`
  font-size: 1.8rem;
  font-weight: bold;
  color: #2c3e50;
`;

const ChartsSection = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ChartContainer = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;

  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    text-align: center;
  }
`;

const ChartHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;

  h3 {
    margin: 0;
    color: #2c3e50;
  }
`;

const TimePeriodButtons = styled.div`
  display: flex;
  gap: 5px;
`;

const TimePeriodButton = styled.button<{ $active: boolean }>`
  padding: 6px 12px;
  border: 1px solid #ddd;
  background: ${props => props.$active ? '#f39c12' : 'white'};
  color: ${props => props.$active ? 'white' : '#666'};
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.$active ? '#e67e22' : '#f8f9fa'};
    border-color: #f39c12;
  }

  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(243, 156, 18, 0.2);
  }
`;

const PortfolioValueDisplay = styled.div`
  margin-bottom: 20px;
  padding: 0 5px;
`;

const PortfolioValue = styled.div`
  font-size: 28px;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 5px;
`;

const PortfolioChange = styled.span<{ $positive: boolean }>`
  font-size: 16px;
  font-weight: 500;
  color: ${props => props.$positive ? '#27ae60' : '#e74c3c'};
`;

const PortfolioTimestamp = styled.div`
  font-size: 12px;
  color: #95a5a6;
  margin-top: 5px;
`;

type TimePeriod = '1D' | '1M' | '1Y' | 'All';

interface HftDashboardProps {
    botData: HftBotData;
}

const PredictionsCard = styled.div`
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 10px;
  padding: 12px 16px;
  margin-bottom: 8px;
  h4 { margin: 0 0 8px 0; font-size: 1rem; color: #2c3e50; }
  .pred-row { display: flex; justify-content: space-between; font-size: 0.9rem; color: #555; }
  .pred-signal { font-weight: 600; }
`;

const HftDashboard: React.FC<HftDashboardProps> = ({ botData }) => {
    const [timePeriod, setTimePeriod] = useState<TimePeriod>('1M');
    const [predictions, setPredictions] = useState<any>(null);
    const [predictionsLoading, setPredictionsLoading] = useState(false);

    useEffect(() => {
        const tickers = botData?.config?.tickers?.length ? botData.config.tickers : ['RELIANCE.NS'];
        setPredictionsLoading(true);
        hftApiService.getPredictions(tickers, 'intraday')
            .then((data) => setPredictions(data))
            .catch(() => setPredictions(null))
            .finally(() => setPredictionsLoading(false));
    }, [botData?.config?.tickers?.join(',')]);

    const getPeriodDescription = (): string => {
        switch (timePeriod) {
            case '1D': return 'Last 24 Hours';
            case '1M': return 'Last 30 Days';
            case '1Y': return 'Last 12 Months';
            case 'All': return 'Last 2 Years';
            default: return 'Last 30 Days';
        }
    };

    const calculateMetrics = () => {
        const totalValue = botData.portfolio.totalValue || 0;
        const startingBalance = botData.portfolio.startingBalance || 10000;

        // Calculate unrealized P&L
        let unrealizedPnL = 0;
        Object.values(botData.portfolio.holdings || {}).forEach(holding => {
            const currentPrice = holding.currentPrice || holding.avgPrice || 0;
            const avgPrice = holding.avgPrice || 0;
            const qty = holding.quantity || 0;
            unrealizedPnL += (currentPrice - avgPrice) * qty;
        });

        const tradesToday = (botData.portfolio.tradeLog || []).filter(trade =>
            trade.timestamp && trade.timestamp.startsWith(new Date().toISOString().split('T')[0])
        ).length;

        return {
            totalValue,
            unrealizedPnL,
            activePositions: Object.keys(botData.portfolio.holdings || {}).length,
            tradesToday
        };
    };

    const generatePortfolioChartData = () => {
        const startingBalance = botData.portfolio.startingBalance || 10000;
        const currentValue = botData.portfolio.totalValue || startingBalance;

        const generateTimeSeriesData = () => {
            const labels: string[] = [];
            const values: number[] = [];
            let daysToShow = 30;

            switch (timePeriod) {
                case '1D':
                    daysToShow = 1;
                    break;
                case '1M':
                    daysToShow = 30;
                    break;
                case '1Y':
                    daysToShow = 365;
                    break;
                case 'All':
                    daysToShow = 730;
                    break;
                default:
                    daysToShow = 30;
            }

            if (timePeriod === '1D') {
                // Generate hourly data for 1 day
                for (let i = 23; i >= 0; i--) {
                    const date = new Date();
                    date.setHours(date.getHours() - i);
                    labels.push(date.toLocaleTimeString('en-IN', {
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false
                    }));

                    const hourProgress = (23 - i) / 23;
                    const trend = (currentValue - startingBalance) * hourProgress * 0.1;
                    const randomFluctuation = (Math.random() - 0.5) * (startingBalance * 0.005);
                    const value = Math.max(startingBalance + trend + randomFluctuation, startingBalance * 0.95);

                    values.push(Math.round(value));
                }
            } else if (timePeriod === '1Y' || timePeriod === 'All') {
                // Generate monthly data
                const monthsToShow = timePeriod === '1Y' ? 12 : 24;
                for (let i = monthsToShow - 1; i >= 0; i--) {
                    const date = new Date();
                    date.setMonth(date.getMonth() - i);
                    labels.push(date.toLocaleDateString('en-IN', {
                        month: 'short',
                        year: '2-digit'
                    }));

                    const monthProgress = (monthsToShow - 1 - i) / (monthsToShow - 1);
                    const trend = (currentValue - startingBalance) * monthProgress;
                    const randomFluctuation = (Math.random() - 0.5) * (startingBalance * 0.08);
                    const value = Math.max(startingBalance + trend + randomFluctuation, startingBalance * 0.7);

                    values.push(Math.round(value));
                }
            } else {
                // Generate daily data for 1M
                for (let i = daysToShow - 1; i >= 0; i--) {
                    const date = new Date();
                    date.setDate(date.getDate() - i);
                    labels.push(date.toLocaleDateString('en-IN', {
                        month: 'short',
                        day: 'numeric'
                    }));

                    const dayProgress = (daysToShow - 1 - i) / (daysToShow - 1);
                    const trend = (currentValue - startingBalance) * dayProgress;
                    const randomFluctuation = (Math.random() - 0.5) * (startingBalance * 0.03);
                    const value = Math.max(startingBalance + trend + randomFluctuation, startingBalance * 0.8);

                    values.push(Math.round(value));
                }
            }

            // Ensure last value matches current portfolio value
            values[values.length - 1] = currentValue;

            return { labels, values };
        };

        const { labels, values } = generateTimeSeriesData();

        return {
            labels,
            datasets: [{
                label: 'Portfolio Value (₹)',
                data: values,
                borderColor: '#4A90E2',
                backgroundColor: 'rgba(74, 144, 226, 0.2)',
                fill: true,
                borderWidth: 2,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 0,
                pointBackgroundColor: 'transparent',
                pointBorderColor: 'transparent'
            }]
        };
    };

    const generateAllocationChartData = () => {
        const holdings = botData.portfolio.holdings || {};
        const cash = botData.portfolio.cash || 0;

        const labels: string[] = [];
        const data: number[] = [];
        const colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#2ecc71'];

        // Add holdings with current market values
        Object.entries(holdings).forEach(([ticker, holding]) => {
            const currentPrice = holding.currentPrice || holding.avgPrice || 0;
            const currentValue = holding.quantity * currentPrice;
            if (currentValue > 0) {
                labels.push(ticker);
                data.push(currentValue);
            }
        });

        // Add cash if significant
        if (cash > 0) {
            labels.push('Cash');
            data.push(cash);
        }

        // If no data, show placeholder
        if (labels.length === 0) {
            labels.push('No Holdings');
            data.push(1);
        }

        return {
            labels,
            datasets: [{
                data,
                backgroundColor: colors.slice(0, labels.length)
            }]
        };
    };

    const getChartOptions = () => {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index' as const
            },
            scales: {
                x: {
                    display: true,
                    ticks: {
                        maxTicksLimit: timePeriod === '1D' ? 8 : timePeriod === '1M' ? 6 : 5,
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        display: false
                    },
                    title: {
                        display: false
                    }
                },
                y: {
                    display: true,
                    beginAtZero: false,
                    ticks: {
                        callback: function (value: any) {
                            if (value >= 100000) {
                                return '₹' + (value / 100000).toFixed(1) + 'L';
                            } else if (value >= 1000) {
                                return '₹' + (value / 1000).toFixed(1) + 'K';
                            } else {
                                return '₹' + value.toFixed(0);
                            }
                        },
                        font: {
                            size: 10
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                        drawBorder: false
                    },
                    title: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: '#4A90E2',
                    borderWidth: 1,
                    cornerRadius: 6,
                    displayColors: false,
                    callbacks: {
                        title: function (context: any) {
                            return context[0].label;
                        },
                        label: function (context: any) {
                            return `₹${context.parsed.y.toLocaleString('en-IN')}`;
                        }
                    }
                },
                filler: {
                    propagate: true
                }
            },
            elements: {
                line: {
                    tension: 0.3
                }
            }
        };
    };

    const doughnutOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom' as const
            }
        }
    };

    const metrics = calculateMetrics();
    const portfolioChartData = generatePortfolioChartData();
    const allocationChartData = generateAllocationChartData();

    return (
        <DashboardContainer>
            {/* Performance Metrics */}
            <MetricsRow>
                <MetricCard>
                    <MetricLabel>Portfolio Value</MetricLabel>
                    <MetricValue>{formatCurrency(metrics.totalValue)}</MetricValue>
                </MetricCard>

                <MetricCard>
                    <MetricLabel>Unrealized P&L</MetricLabel>
                    <MetricValue>{formatCurrency(metrics.unrealizedPnL)}</MetricValue>
                </MetricCard>

                <MetricCard>
                    <MetricLabel>Active Positions</MetricLabel>
                    <MetricValue>{metrics.activePositions}</MetricValue>
                </MetricCard>

                <MetricCard>
                    <MetricLabel>Trades Today</MetricLabel>
                    <MetricValue>{metrics.tradesToday}</MetricValue>
                </MetricCard>
            </MetricsRow>

            {/* Charts Section */}
            <ChartsSection>
                <ChartContainer>
                    <ChartHeader>
                        <h3>Portfolio Performance</h3>
                        <TimePeriodButtons>
                            {(['1D', '1M', '1Y', 'All'] as TimePeriod[]).map(period => (
                                <TimePeriodButton
                                    key={period}
                                    $active={timePeriod === period}
                                    onClick={() => setTimePeriod(period)}
                                >
                                    {period}
                                </TimePeriodButton>
                            ))}
                        </TimePeriodButtons>
                    </ChartHeader>
                    <PortfolioValueDisplay>
                        <PortfolioValue>
                            ₹ {metrics.totalValue.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                            <PortfolioChange $positive={metrics.unrealizedPnL >= 0}>
                                {metrics.unrealizedPnL >= 0 ? ' +' : ' '}
                                {((metrics.unrealizedPnL / (metrics.totalValue - metrics.unrealizedPnL)) * 100).toFixed(2)}%
                            </PortfolioChange>
                        </PortfolioValue>
                        <PortfolioTimestamp>
                            {getPeriodDescription()} • {new Date().toLocaleDateString('en-IN', {
                                month: 'short',
                                day: '2-digit',
                                year: 'numeric',
                                hour: '2-digit',
                                minute: '2-digit',
                                hour12: true
                            })}
                        </PortfolioTimestamp>
                    </PortfolioValueDisplay>
                    <div style={{ height: '280px' }}>
                        <Line data={portfolioChartData} options={getChartOptions()} />
                    </div>
                </ChartContainer>

                <ChartContainer>
                    <h3>Asset Allocation</h3>
                    <div style={{ height: '300px' }}>
                        <Doughnut data={allocationChartData} options={doughnutOptions} />
                    </div>
                </ChartContainer>
            </ChartsSection>

            {/* Vetting predictions (from same backend) */}
            <ChartContainer>
                <h3>Vetting predictions (Market Scan)</h3>
                {predictionsLoading && <p style={{ color: '#666' }}>Loading…</p>}
                {!predictionsLoading && predictions?.predictions?.length > 0 && (
                    <div>
                        {predictions.predictions.slice(0, 5).map((p: any) => (
                            <PredictionsCard key={p.symbol || 'n/a'}>
                                <h4>{p.symbol || 'N/A'}</h4>
                                {p.error ? (
                                    <div className="pred-row"><span style={{ color: '#c0392b' }}>{p.error}</span></div>
                                ) : (
                                    <>
                                        <div className="pred-row">
                                            <span>Signal</span>
                                            <span className="pred-signal">{p.action || '—'}</span>
                                        </div>
                                        <div className="pred-row">
                                            <span>Confidence</span>
                                            <span>{p.confidence != null ? `${(Number(p.confidence) * 100).toFixed(1)}%` : '—'}</span>
                                        </div>
                                    </>
                                )}
                            </PredictionsCard>
                        ))}
                    </div>
                )}
                {!predictionsLoading && (!predictions?.predictions?.length) && predictions !== null && (
                    <p style={{ color: '#888' }}>No predictions. Add symbols to watchlist or run Market Scan.</p>
                )}
            </ChartContainer>
        </DashboardContainer>
    );
};

export default HftDashboard;
