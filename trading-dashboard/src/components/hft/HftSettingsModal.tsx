import React, { useState, useEffect } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { useAuth } from '../../contexts/AuthContext';
import { X, CheckCircle2, AlertCircle, Link2, RefreshCw } from 'lucide-react';
import { hftApiService } from '../../services/hftApiService';
import type { HftSettingsUpdate } from '../../types/hft';

interface SettingsFormData {
    mode: 'paper' | 'live';
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CUSTOM';
    maxAllocation: number | string;
    stopLossPct: number | string;
}

interface HftSettingsModalProps {
    settings: any;
    onSave: (settings: any) => Promise<void>;
    onClose: () => void;
}

const HftSettingsModal: React.FC<HftSettingsModalProps> = ({ settings, onSave, onClose }) => {
    const { theme } = useTheme();
    const { user } = useAuth();
    const isLight = theme === 'light';
    const isSpace = theme === 'space';

    const [formData, setFormData] = useState<SettingsFormData>({
        mode: 'paper',
        riskLevel: 'MEDIUM',
        maxAllocation: 25,
        stopLossPct: 5
    });
    const [loading, setLoading] = useState(false);
    const [dhanConfigured, setDhanConfigured] = useState<boolean | null>(null);
    const [dhanError, setDhanError] = useState<string | null>(null);

    // Per-user demat (link/refresh)
    const [dematLinked, setDematLinked] = useState(false);
    const [dematBroker, setDematBroker] = useState<string>('');
    const [dematMaskedId, setDematMaskedId] = useState<string | null>(null);
    const [dematLoading, setDematLoading] = useState(false);
    const [dematError, setDematError] = useState<string | null>(null);
    const [dematSuccess, setDematSuccess] = useState<string | null>(null);
    const [showDematLinkForm, setShowDematLinkForm] = useState(false);
    const [showDematRefreshForm, setShowDematRefreshForm] = useState(false);
    const [dematLinkForm, setDematLinkForm] = useState({ broker: 'dhan', clientId: '', accessToken: '' });
    const [dematRefreshToken, setDematRefreshToken] = useState('');

    useEffect(() => {
        if (settings) {
            setFormData({
                mode: settings.mode || 'paper',
                riskLevel: settings.riskLevel || 'MEDIUM',
                maxAllocation: settings.maxAllocation ? (settings.maxAllocation * 100) : 25,
                stopLossPct: settings.stopLossPct || 5
            });
        }
    }, [settings]);

    useEffect(() => {
        let cancelled = false;
        hftApiService.getLiveStatus()
            .then((res) => {
                if (!cancelled) {
                    setDhanConfigured(res.dhan_configured ?? false);
                    setDhanError(res.dhan_error ?? null);
                }
            })
            .catch(() => { if (!cancelled) { setDhanConfigured(false); setDhanError(null); } });
        return () => { cancelled = true; };
    }, []);

    useEffect(() => {
        if (!user?.username) {
            setDematLinked(false);
            setDematBroker('');
            setDematMaskedId(null);
            return;
        }
        let cancelled = false;
        hftApiService.getDematStatus()
            .then((res) => {
                if (!cancelled) {
                    setDematLinked(res.linked ?? false);
                    setDematBroker(res.broker ?? '');
                    setDematMaskedId(res.client_id_masked ?? null);
                }
            })
            .catch(() => { if (!cancelled) { setDematLinked(false); setDematBroker(''); setDematMaskedId(null); } });
        return () => { cancelled = true; };
    }, [user?.username]);

    const handleInputChange = (field: keyof SettingsFormData, value: any) => {
        setFormData(prev => {
            const newData = {
                ...prev,
                [field]: value
            };

            // Auto-update stop loss and allocation based on risk level
            if (field === 'riskLevel') {
                if (value === 'CUSTOM') {
                    newData.stopLossPct = '';
                    newData.maxAllocation = '';
                } else {
                    const riskSettings: Record<string, { stopLoss: number; allocation: number }> = {
                        'LOW': { stopLoss: 3, allocation: 15 },
                        'MEDIUM': { stopLoss: 5, allocation: 25 },
                        'HIGH': { stopLoss: 8, allocation: 35 }
                    };

                    if (riskSettings[value]) {
                        newData.stopLossPct = riskSettings[value].stopLoss;
                        newData.maxAllocation = riskSettings[value].allocation;
                    }
                }
            }

            return newData;
        });
    };

    const handleSave = async () => {
        setLoading(true);
        try {
            const maxAllocationNum = parseFloat(String(formData.maxAllocation)) || 0;
            const stopLossPctNum = parseFloat(String(formData.stopLossPct)) || 0;

            // Validate custom mode
            if (formData.riskLevel === 'CUSTOM') {
                if (!formData.maxAllocation || !formData.stopLossPct || maxAllocationNum <= 0 || stopLossPctNum <= 0) {
                    alert('Please enter valid values for both Max Allocation (1-100) and Stop Loss Percentage (1-20) when using Custom risk level.');
                    setLoading(false);
                    return;
                }

                if (maxAllocationNum < 1 || maxAllocationNum > 100) {
                    alert('Max Allocation must be between 1 and 100.');
                    setLoading(false);
                    return;
                }

                if (stopLossPctNum < 1 || stopLossPctNum > 20) {
                    alert('Stop Loss Percentage must be between 1 and 20.');
                    setLoading(false);
                    return;
                }
            }

            const settingsToSave: any = {
                mode: formData.mode,
                riskLevel: formData.riskLevel,
                maxAllocation: maxAllocationNum / 100,
                ...(formData.riskLevel === 'CUSTOM' && { stopLoss: stopLossPctNum / 100 })
            };

            console.log('Saving settings:', settingsToSave);
            await onSave(settingsToSave);
        } catch (error) {
            console.error('Error saving settings:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    const handleSaveDematLink = async () => {
        if (!dematLinkForm.clientId.trim() || !dematLinkForm.accessToken.trim()) {
            setDematError('Client ID and Access Token are required.');
            return;
        }
        setDematLoading(true);
        setDematError(null);
        setDematSuccess(null);
        try {
            await hftApiService.saveDemat(dematLinkForm.broker, dematLinkForm.clientId.trim(), dematLinkForm.accessToken.trim());
            setDematLinked(true);
            setDematBroker(dematLinkForm.broker);
            setDematMaskedId(dematLinkForm.clientId.trim().slice(0, 4) + '***');
            setDematSuccess('Demat account linked.');
            setShowDematLinkForm(false);
            setDematLinkForm(prev => ({ ...prev, clientId: '', accessToken: '' }));
        } catch (err: any) {
            setDematError(err?.response?.data?.detail || err?.message || 'Failed to save demat credentials.');
        } finally {
            setDematLoading(false);
        }
    };

    const handleRefreshDematToken = async () => {
        if (!dematRefreshToken.trim()) {
            setDematError('Enter the new access token.');
            return;
        }
        setDematLoading(true);
        setDematError(null);
        setDematSuccess(null);
        try {
            await hftApiService.refreshDematToken(dematRefreshToken.trim());
            setDematSuccess('Access token updated. Use it for the next 24h.');
            setShowDematRefreshForm(false);
            setDematRefreshToken('');
        } catch (err: any) {
            setDematError(err?.response?.data?.detail || err?.message || 'Failed to update token.');
        } finally {
            setDematLoading(false);
        }
    };

    const modalBg = isLight ? 'bg-white' : isSpace ? 'bg-slate-800/95' : 'bg-slate-800';
    const modalBorder = isLight ? 'border-gray-200' : isSpace ? 'border-purple-900/30' : 'border-slate-700';
    const textPrimary = isLight ? 'text-gray-900' : 'text-white';
    const textMuted = isLight ? 'text-gray-600' : 'text-gray-400';
    const inputBg = isLight ? 'bg-white' : 'bg-slate-700';
    const inputBorder = isLight ? 'border-gray-300' : 'border-slate-600';
    const inputDisabledBg = isLight ? 'bg-gray-100' : 'bg-slate-800';
    const selectBg = isLight ? 'bg-white' : 'bg-slate-700';
    const selectText = isLight ? 'text-gray-900' : 'text-white';

    return (
        <div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={handleOverlayClick}
        >
            <div
                className={`${modalBg} border ${modalBorder} rounded-xl w-full max-w-lg shadow-2xl max-h-[90vh] overflow-y-auto`}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header */}
                <div className={`flex items-center justify-between p-6 border-b ${modalBorder}`}>
                    <h3 className={`text-xl font-bold ${textPrimary}`}>Settings</h3>
                    <button
                        onClick={onClose}
                        className={`p-2 rounded-lg transition-colors ${
                            isLight
                                ? 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                                : 'text-gray-400 hover:bg-slate-700 hover:text-white'
                        }`}
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-6">
                    {/* Trading Mode */}
                    <div>
                        <label className={`block text-sm font-semibold mb-2 ${textPrimary}`}>
                            Trading Mode:
                        </label>
                        <select
                            value={formData.mode}
                            onChange={(e) => handleInputChange('mode', e.target.value as 'paper' | 'live')}
                            disabled={loading}
                            className={`w-full px-4 py-3 rounded-lg border-2 transition-colors ${selectBg} ${selectText} ${inputBorder} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed`}
                            style={{
                                appearance: 'auto',
                                WebkitAppearance: 'menulist',
                                MozAppearance: 'menulist'
                            }}
                        >
                            <option value="paper" className={selectText}>Paper Trading</option>
                            <option value="live" className={selectText}>Live Trading</option>
                        </select>
                        <p className={`text-xs mt-2 ${textMuted}`}>
                            {formData.mode === 'live'
                                ? 'Live: real positions and prices from Dhan. Set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID in backend env (e.g. Render).'
                                : 'Paper: no positions shown (no stale data).'}
                        </p>
                        {formData.mode === 'live' && (
                            <div className={`flex items-center gap-2 mt-2 px-3 py-2 rounded-lg text-sm ${dhanConfigured === null ? textMuted : dhanConfigured ? 'bg-green-500/10 text-green-600 dark:text-green-400' : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'}`}>
                                {dhanConfigured === null ? (
                                    <span>Checking Dhan credentials…</span>
                                ) : dhanConfigured ? (
                                    <>
                                        <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                                        <span>Dhan credentials: Configured</span>
                                    </>
                                ) : (
                                    <>
                                        <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                        <span>Dhan credentials: Not configured. Set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID in backend environment.</span>
                                    </>
                                )}
                            </div>
                        )}
                        {formData.mode === 'live' && dhanError && (
                            <div className="mt-2 px-3 py-2 rounded-lg text-sm bg-amber-500/10 text-amber-600 dark:text-amber-400">
                                <span className="font-medium">Fetch error: </span>
                                <span>{dhanError}</span>
                            </div>
                        )}
                    </div>

                    {/* Demat account (per-user) */}
                    <div className={`border-t pt-4 ${modalBorder}`}>
                        <label className={`block text-sm font-semibold mb-2 ${textPrimary}`}>
                            <Link2 className="w-4 h-4 inline-block mr-1 align-middle" /> Demat account
                        </label>
                        {!user?.username ? (
                            <p className={`text-sm ${textMuted}`}>Log in to link your demat (Client ID + Access Token). Portfolio and orders will use your account only.</p>
                        ) : (
                            <>
                                {dematError && (
                                    <div className="mb-2 px-3 py-2 rounded-lg text-sm bg-red-500/10 text-red-600 dark:text-red-400">{dematError}</div>
                                )}
                                {dematSuccess && (
                                    <div className="mb-2 px-3 py-2 rounded-lg text-sm bg-green-500/10 text-green-600 dark:text-green-400">{dematSuccess}</div>
                                )}
                                {dematLinked ? (
                                    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm bg-green-500/10 text-green-600 dark:text-green-400`}>
                                        <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                                        <span>Linked</span>
                                        {dematBroker && <span>({dematBroker})</span>}
                                        {dematMaskedId && <span> • {dematMaskedId}</span>}
                                    </div>
                                ) : (
                                    <p className={`text-sm ${textMuted}`}>No demat linked. Add credentials to see your portfolio and place orders.</p>
                                )}
                                {!showDematLinkForm && !showDematRefreshForm && (
                                    <div className="flex gap-2 mt-2">
                                        <button
                                            type="button"
                                            onClick={() => { setShowDematLinkForm(true); setShowDematRefreshForm(false); setDematError(null); setDematSuccess(null); }}
                                            disabled={dematLoading}
                                            className={`px-3 py-1.5 rounded-lg text-sm font-medium ${dematLinked ? 'border border-current' : 'bg-blue-600 text-white'} ${isLight ? 'text-blue-600 border-blue-600 hover:bg-blue-50' : 'border-slate-400 hover:bg-slate-700'}`}
                                        >
                                            {dematLinked ? 'Update credentials' : 'Link demat'}
                                        </button>
                                        {dematLinked && (
                                            <button
                                                type="button"
                                                onClick={() => { setShowDematRefreshForm(true); setShowDematLinkForm(false); setDematError(null); setDematSuccess(null); }}
                                                disabled={dematLoading}
                                                className={`px-3 py-1.5 rounded-lg text-sm font-medium flex items-center gap-1 ${isLight ? 'text-amber-600 border border-amber-600 hover:bg-amber-50' : 'border border-amber-400 text-amber-400 hover:bg-slate-700'}`}
                                            >
                                                <RefreshCw className="w-3.5 h-3.5" /> Refresh token
                                            </button>
                                        )}
                                    </div>
                                )}
                                {showDematLinkForm && (
                                    <div className={`mt-3 p-3 rounded-lg border ${inputBorder} ${inputBg}`}>
                                        <p className={`text-xs ${textMuted} mb-2`}>Broker, Client ID and Access Token are stored only for your account and used for portfolio and orders.</p>
                                        <select
                                            value={dematLinkForm.broker}
                                            onChange={(e) => setDematLinkForm(prev => ({ ...prev, broker: e.target.value }))}
                                            className={`w-full mb-2 px-3 py-2 rounded border ${selectBg} ${selectText} ${inputBorder} text-sm`}
                                        >
                                            <option value="dhan">Dhan</option>
                                        </select>
                                        <input
                                            type="text"
                                            placeholder="Client ID"
                                            value={dematLinkForm.clientId}
                                            onChange={(e) => setDematLinkForm(prev => ({ ...prev, clientId: e.target.value }))}
                                            className={`w-full mb-2 px-3 py-2 rounded border ${inputBg} ${inputBorder} text-sm`}
                                        />
                                        <input
                                            type="password"
                                            placeholder="Access Token"
                                            value={dematLinkForm.accessToken}
                                            onChange={(e) => setDematLinkForm(prev => ({ ...prev, accessToken: e.target.value }))}
                                            className={`w-full mb-2 px-3 py-2 rounded border ${inputBg} ${inputBorder} text-sm`}
                                        />
                                        <div className="flex gap-2">
                                            <button type="button" onClick={() => setShowDematLinkForm(false)} className="px-3 py-1.5 rounded text-sm border border-current">Cancel</button>
                                            <button type="button" onClick={handleSaveDematLink} disabled={dematLoading} className="px-3 py-1.5 rounded text-sm bg-green-600 text-white disabled:opacity-50">{dematLoading ? 'Saving...' : 'Save'}</button>
                                        </div>
                                    </div>
                                )}
                                {showDematRefreshForm && (
                                    <div className={`mt-3 p-3 rounded-lg border ${inputBorder} ${inputBg}`}>
                                        <p className={`text-xs ${textMuted} mb-2`}>Access tokens often expire every 24h. Paste the new token below.</p>
                                        <input
                                            type="password"
                                            placeholder="New Access Token"
                                            value={dematRefreshToken}
                                            onChange={(e) => setDematRefreshToken(e.target.value)}
                                            className={`w-full mb-2 px-3 py-2 rounded border ${inputBg} ${inputBorder} text-sm`}
                                        />
                                        <div className="flex gap-2">
                                            <button type="button" onClick={() => setShowDematRefreshForm(false)} className="px-3 py-1.5 rounded text-sm border border-current">Cancel</button>
                                            <button type="button" onClick={handleRefreshDematToken} disabled={dematLoading} className="px-3 py-1.5 rounded text-sm bg-amber-600 text-white disabled:opacity-50">{dematLoading ? 'Updating...' : 'Update token'}</button>
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                    {/* Risk Level */}
                    <div>
                        <label className={`block text-sm font-semibold mb-2 ${textPrimary}`}>
                            Risk Level:
                        </label>
                        <select
                            value={formData.riskLevel}
                            onChange={(e) => handleInputChange('riskLevel', e.target.value)}
                            disabled={loading}
                            className={`w-full px-4 py-3 rounded-lg border-2 transition-colors ${selectBg} ${selectText} ${inputBorder} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed`}
                            style={{
                                appearance: 'auto',
                                WebkitAppearance: 'menulist',
                                MozAppearance: 'menulist'
                            }}
                        >
                            <option value="LOW" className={selectText}>Low (3% stop-loss, 15% allocation)</option>
                            <option value="MEDIUM" className={selectText}>Medium (5% stop-loss, 25% allocation)</option>
                            <option value="HIGH" className={selectText}>High (8% stop-loss, 35% allocation)</option>
                            <option value="CUSTOM" className={selectText}>Custom (Set your own values)</option>
                        </select>
                    </div>

                    {/* Max Allocation */}
                    <div>
                        <label className={`block text-sm font-semibold mb-2 ${textPrimary}`}>
                            Max Allocation per Trade (%):
                        </label>
                        <input
                            type="number"
                            min="1"
                            max="100"
                            value={formData.maxAllocation}
                            placeholder={formData.riskLevel === 'CUSTOM' ? 'Enter percentage (1-100)' : ''}
                            onChange={(e) => handleInputChange('maxAllocation', e.target.value)}
                            disabled={loading || formData.riskLevel !== 'CUSTOM'}
                            className={`w-full px-4 py-3 rounded-lg border-2 transition-colors ${
                                formData.riskLevel === 'CUSTOM' ? inputBg : inputDisabledBg
                            } ${selectText} ${inputBorder} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed`}
                        />
                        {formData.riskLevel !== 'CUSTOM' && (
                            <p className={`text-xs mt-2 ${textMuted}`}>
                                Select "Custom" risk level to modify this value
                            </p>
                        )}
                    </div>

                    {/* Stop Loss */}
                    <div>
                        <label className={`block text-sm font-semibold mb-2 ${textPrimary}`}>
                            Stop Loss Percentage (%):
                        </label>
                        <input
                            type="number"
                            min="1"
                            max="20"
                            step="0.1"
                            value={formData.stopLossPct}
                            placeholder={formData.riskLevel === 'CUSTOM' ? 'Enter percentage (1-20)' : ''}
                            onChange={(e) => handleInputChange('stopLossPct', e.target.value)}
                            disabled={loading || formData.riskLevel !== 'CUSTOM'}
                            className={`w-full px-4 py-3 rounded-lg border-2 transition-colors ${
                                formData.riskLevel === 'CUSTOM' ? inputBg : inputDisabledBg
                            } ${selectText} ${inputBorder} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed`}
                        />
                        {formData.riskLevel !== 'CUSTOM' && (
                            <p className={`text-xs mt-2 ${textMuted}`}>
                                Select "Custom" risk level to modify this value
                            </p>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className={`flex items-center justify-end gap-3 p-6 border-t ${modalBorder}`}>
                    <button
                        onClick={onClose}
                        disabled={loading}
                        className={`px-6 py-2.5 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                            isLight
                                ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                                : 'bg-slate-700 text-gray-200 hover:bg-slate-600'
                        }`}
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={loading}
                        className="px-6 py-2.5 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-semibold transition-colors"
                    >
                        {loading ? 'Saving...' : 'Save Settings'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default HftSettingsModal;
