import React, { useState, useEffect, useRef, useCallback } from 'react';
import { config } from '../../config';

// ─── Types ────────────────────────────────────────────────────────────────────

interface IndicatorEvent {
    name: string;
    value: string | number;
    signal: string; // 'bullish'|'bearish'|'neutral'|'oversold'|'overbought'|'high'|'low'|'normal'|'positive'|'negative'
}

interface AnalysisResult {
    symbol: string;
    recommendation: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    reasoning: string;
    risk_score: number;
    target_price: number | null;
    stop_loss: number | null;
    sentiment: string;
    sentiment_score: number;
    indicators: Record<string, { value?: string | number; signal?: string;[k: string]: unknown }>;
    timestamp: string;
}

interface LogLine { level: string; message: string; }

interface AnalysisPanelProps {
    symbol: string;
    active: boolean;          // true = bot is running, false = idle
    onResult?: (result: AnalysisResult) => void;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const SIGNAL_COLORS: Record<string, { bg: string; border: string; text: string; dot: string }> = {
    BUY: { bg: 'rgba(39,174,96,0.12)', border: '#27ae60', text: '#27ae60', dot: '#27ae60' },
    SELL: { bg: 'rgba(231,76,60,0.12)', border: '#e74c3c', text: '#e74c3c', dot: '#e74c3c' },
    HOLD: { bg: 'rgba(243,156,18,0.12)', border: '#f39c12', text: '#f39c12', dot: '#f39c12' },
};

const IND_COLOR: Record<string, string> = {
    bullish: '#27ae60', oversold: '#27ae60', positive: '#27ae60', high: '#27ae60',
    bearish: '#e74c3c', overbought: '#e74c3c', negative: '#e74c3c',
    neutral: '#95a5a6', normal: '#95a5a6', low: '#f39c12',
};

const LOG_COLOR: Record<string, string> = {
    ERROR: '#ff7b72', WARNING: '#e3b341', INFO: '#79c0ff', default: '#8b949e',
};

function indSignalColor(sig: string) { return IND_COLOR[sig] ?? '#95a5a6'; }
function indSignalLabel(sig: string) {
    const map: Record<string, string> = {
        bullish: '▲ Bullish', bearish: '▼ Bearish', neutral: '◆ Neutral',
        oversold: '▲ Oversold', overbought: '▼ Overbought',
        positive: '▲ Positive', negative: '▼ Negative',
        high: '▲ High Vol', low: '▼ Low Vol', normal: '◆ Normal Vol',
    };
    return map[sig] ?? sig;
}

// ─── Component ────────────────────────────────────────────────────────────────

const HftAnalysisPanel: React.FC<AnalysisPanelProps> = ({ symbol, active, onResult }) => {
    const [progress, setProgress] = useState<{ step: string; pct: number } | null>(null);
    const [indicators, setIndicators] = useState<IndicatorEvent[]>([]);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [logs, setLogs] = useState<LogLine[]>([]);
    const [done, setDone] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showLogs, setShowLogs] = useState(false);
    const [connecting, setConnecting] = useState(false);

    const esRef = useRef<EventSource | null>(null);
    const logRef = useRef<HTMLDivElement | null>(null);
    const startRef = useRef(false);

    // Auto-scroll logs
    useEffect(() => {
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, [logs]);

    const resetState = useCallback(() => {
        setProgress(null);
        setIndicators([]);
        setResult(null);
        setLogs([]);
        setDone(false);
        setError(null);
        setConnecting(true);
    }, []);

    const startStream = useCallback(() => {
        if (esRef.current) { esRef.current.close(); esRef.current = null; }
        resetState();

        const url = `${config.API_BASE_URL}/api/analyze-stream?symbol=${encodeURIComponent(symbol)}`;
        const es = new EventSource(url);
        esRef.current = es;

        es.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                switch (msg.type) {
                    case 'connected':
                        setConnecting(false);
                        break;
                    case 'progress':
                        setProgress({ step: msg.step, pct: msg.pct });
                        break;
                    case 'log':
                        setLogs(prev => [...prev.slice(-400), { level: msg.level ?? 'INFO', message: msg.message }]);
                        break;
                    case 'indicator':
                        setIndicators(prev => {
                            const next = prev.filter(i => i.name !== msg.name);
                            return [...next, { name: msg.name, value: msg.value, signal: msg.signal }];
                        });
                        break;
                    case 'result':
                        setResult(msg.data);
                        if (onResult) onResult(msg.data);
                        break;
                    case 'error':
                        setError(msg.message);
                        break;
                    case 'done':
                        setDone(true);
                        setProgress(p => p ? { ...p, pct: 100, step: 'Analysis complete' } : p);
                        es.close();
                        esRef.current = null;
                        break;
                }
            } catch { /* malformed event */ }
        };

        es.onerror = () => {
            setConnecting(false);
            if (!done) setError('Stream connection lost. The server may still be processing.');
            es.close();
            esRef.current = null;
        };
    }, [symbol, onResult, resetState, done]);

    // Start when active flips true (bot started)
    useEffect(() => {
        if (active && !startRef.current) {
            startRef.current = true;
            startStream();
        }
        if (!active) {
            startRef.current = false;
        }
        return () => {
            if (esRef.current) { esRef.current.close(); esRef.current = null; }
        };
    }, [active, startStream]);

    const rec = result?.recommendation ?? 'HOLD';
    const colors = SIGNAL_COLORS[rec] ?? SIGNAL_COLORS.HOLD;

    return (
        <div style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 14,
            padding: 20,
            marginBottom: 16,
            fontFamily: 'Inter, system-ui, sans-serif',
        }}>
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                <span style={{
                    fontSize: '1rem', fontWeight: 700, letterSpacing: '0.04em',
                    color: '#e6edf3', background: 'rgba(255,255,255,0.08)',
                    padding: '3px 10px', borderRadius: 6,
                }}>
                    {symbol.replace('.NS', '').replace('.BO', '')}
                </span>
                <span style={{ fontSize: '0.75rem', color: '#8b949e' }}>{symbol}</span>
                {connecting && (
                    <span style={{ fontSize: '0.75rem', color: '#f39c12', marginLeft: 'auto' }}>
                        ⏳ Connecting…
                    </span>
                )}
                {active && !done && !connecting && (
                    <span style={{ fontSize: '0.75rem', color: '#79c0ff', marginLeft: 'auto', animation: 'pulse 1.5s infinite' }}>
                        ● Live Analysis Running
                    </span>
                )}
                {done && (
                    <span style={{ fontSize: '0.75rem', color: '#3fb950', marginLeft: 'auto' }}>
                        ✓ Analysis Complete
                    </span>
                )}
                {!active && !done && (
                    <span style={{ fontSize: '0.75rem', color: '#8b949e', marginLeft: 'auto' }}>
                        Start bot to analyze
                    </span>
                )}
            </div>

            {/* Progress bar */}
            {progress && !done && (
                <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#8b949e', marginBottom: 6 }}>
                        <span>{progress.step}</span>
                        <span>{progress.pct}%</span>
                    </div>
                    <div style={{ height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{
                            height: '100%', width: `${progress.pct}%`,
                            background: 'linear-gradient(90deg, #4a90e2, #27ae60)',
                            borderRadius: 4,
                            transition: 'width 0.4s ease',
                        }} />
                    </div>
                </div>
            )}

            {/* Error banner */}
            {error && (
                <div style={{
                    background: 'rgba(255,123,114,0.12)', border: '1px solid #ff7b72',
                    borderRadius: 8, padding: '10px 14px', color: '#ff7b72',
                    fontSize: '0.8rem', marginBottom: 12,
                }}>
                    ⚠️ {error}
                </div>
            )}

            {/* Main result card */}
            {result && (
                <div style={{
                    background: colors.bg,
                    border: `1.5px solid ${colors.border}`,
                    borderRadius: 10,
                    padding: 16,
                    marginBottom: 16,
                }}>
                    {/* Signal row */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                        <span style={{
                            fontSize: '1.25rem', fontWeight: 800, color: colors.text,
                            letterSpacing: '0.06em',
                        }}>
                            {rec}
                        </span>
                        <span style={{
                            background: colors.border, color: '#fff',
                            fontSize: '0.72rem', fontWeight: 700,
                            padding: '2px 10px', borderRadius: 30,
                        }}>
                            {(result.confidence * 100).toFixed(1)}% confidence
                        </span>
                        <span style={{
                            marginLeft: 'auto', fontSize: '0.75rem', color: '#8b949e',
                        }}>
                            {new Date(result.timestamp).toLocaleTimeString('en-IN')}
                        </span>
                    </div>

                    {/* Key metrics row */}
                    <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12, fontSize: '0.82rem' }}>
                        {result.target_price != null && (
                            <span style={{ color: '#e6edf3' }}>
                                🎯 Target: <strong style={{ color: '#3fb950' }}>₹{result.target_price}</strong>
                            </span>
                        )}
                        {result.stop_loss != null && (
                            <span style={{ color: '#e6edf3' }}>
                                🛡 Stop Loss: <strong style={{ color: '#ff7b72' }}>₹{result.stop_loss}</strong>
                            </span>
                        )}
                        <span style={{ color: '#e6edf3' }}>
                            ⚡ Risk: <strong>{(result.risk_score * 100).toFixed(0)}%</strong>
                        </span>
                        {result.sentiment && (
                            <span style={{ color: '#e6edf3' }}>
                                📰 Sentiment: <strong style={{ color: indSignalColor(result.sentiment) }}>
                                    {result.sentiment}
                                </strong>
                            </span>
                        )}
                    </div>

                    {/* Reasoning */}
                    {result.reasoning && (
                        <div style={{
                            borderTop: '1px solid rgba(255,255,255,0.08)',
                            paddingTop: 10, fontSize: '0.8rem', color: '#8b949e',
                            lineHeight: 1.5,
                        }}>
                            {result.reasoning}
                        </div>
                    )}
                </div>
            )}

            {/* Indicator mini-cards */}
            {indicators.length > 0 && (
                <div>
                    <div style={{ fontSize: '0.75rem', color: '#8b949e', marginBottom: 8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        Technical Indicators
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 8 }}>
                        {indicators.map((ind) => (
                            <div key={ind.name} style={{
                                background: 'rgba(255,255,255,0.04)',
                                border: `1px solid ${indSignalColor(ind.signal)}40`,
                                borderRadius: 8,
                                padding: '10px 12px',
                            }}>
                                <div style={{ fontSize: '0.7rem', color: '#8b949e', marginBottom: 4 }}>{ind.name}</div>
                                <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#e6edf3', marginBottom: 4 }}>
                                    {typeof ind.value === 'number' ? ind.value.toLocaleString() : ind.value}
                                </div>
                                <div style={{
                                    fontSize: '0.7rem', fontWeight: 600,
                                    color: indSignalColor(ind.signal),
                                }}>
                                    {indSignalLabel(ind.signal)}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Collapsible terminal logs */}
            {logs.length > 0 && (
                <div style={{ marginTop: 14 }}>
                    <button
                        onClick={() => setShowLogs(v => !v)}
                        style={{
                            background: 'none', border: 'none', cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: 6,
                            color: '#8b949e', fontSize: '0.75rem', padding: '4px 0',
                        }}
                    >
                        <span style={{ transition: 'transform 0.2s', transform: showLogs ? 'rotate(90deg)' : 'none' }}>▶</span>
                        {showLogs ? 'Hide' : 'Show'} backend logs ({logs.length} lines)
                    </button>
                    {showLogs && (
                        <div
                            ref={logRef}
                            style={{
                                marginTop: 8,
                                background: '#0d1117',
                                border: '1px solid #30363d',
                                borderRadius: 8,
                                padding: '10px 14px',
                                maxHeight: 220,
                                overflowY: 'auto',
                                fontFamily: "'Consolas', 'Monaco', 'Courier New', monospace",
                                fontSize: '0.7rem',
                                lineHeight: 1.7,
                            }}
                        >
                            {logs.map((l, i) => (
                                <div key={i} style={{ color: LOG_COLOR[l.level] ?? LOG_COLOR.default }}>
                                    {l.message}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Idle placeholder */}
            {!active && !result && !progress && !error && (
                <div style={{
                    textAlign: 'center', padding: '24px 0',
                    color: '#484f58', fontSize: '0.85rem',
                }}>
                    Start the bot to run AI analysis on {symbol}
                </div>
            )}

            <style>{`
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.4; }
                }
            `}</style>
        </div>
    );
};

export default HftAnalysisPanel;
