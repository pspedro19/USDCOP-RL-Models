import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const MetricCard = ({ title, value, trend, trendValue, status }) => {
    // Determine color and icon based on status or trend
    const getTrendIcon = () => {
        if (status === 'positive' || trend > 0) return <TrendingUp size={16} />;
        if (status === 'negative' || trend < 0) return <TrendingDown size={16} />;
        return <Minus size={16} />;
    };

    const traverseStatus = () => {
        if (trend > 0) return 'trend-up';
        if (trend < 0) return 'trend-down';
        return 'trend-neutral';
    }

    return (
        <div className="glass-panel metric-card">
            <div className="metric-title">{title}</div>
            <div className="metric-value">{value}</div>
            {trendValue && (
                <div className={`metric-trend ${traverseStatus()}`}>
                    {getTrendIcon()}
                    <span>{trendValue}</span>
                </div>
            )}
        </div>
    );
};

export default MetricCard;
