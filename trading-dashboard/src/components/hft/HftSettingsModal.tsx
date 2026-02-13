import React, { useState, useEffect } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { X } from 'lucide-react';
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
    const isLight = theme === 'light';
    const isSpace = theme === 'space';

    const [formData, setFormData] = useState<SettingsFormData>({
        mode: 'paper',
        riskLevel: 'MEDIUM',
        maxAllocation: 25,
        stopLossPct: 5
    });
    const [loading, setLoading] = useState(false);

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
                                ? 'Live: real positions and prices from Dhan (run HFT2 backend with env).'
                                : 'Paper: no positions shown (no stale data).'}
                        </p>
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
