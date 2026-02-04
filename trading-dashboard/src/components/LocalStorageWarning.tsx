import { Database, AlertTriangle } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface LocalStorageWarningProps {
  feature: string;
}

export const LocalStorageWarning = ({ feature }: LocalStorageWarningProps) => {
  const { theme } = useTheme();
  const isLight = theme === 'light';

  return (
    <div className={`border-l-4 rounded-lg p-4 mb-6 ${
      isLight 
        ? 'bg-amber-50 border-amber-500 text-amber-900' 
        : 'bg-amber-900/20 border-amber-500 text-amber-200'
    }`}>
      <div className="flex items-start gap-3">
        <Database className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
          isLight ? 'text-amber-600' : 'text-amber-400'
        }`} />
        <div className="flex-1">
          <p className="font-semibold text-sm mb-1">
            Frontend-Only Mode
          </p>
          <p className="text-xs leading-relaxed">
            {feature} data is stored locally in your browser. Backend does not support this feature. 
            Data will be lost if you clear browser storage.
          </p>
        </div>
      </div>
    </div>
  );
};
