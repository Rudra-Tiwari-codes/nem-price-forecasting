'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';

const REGIONS = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1'];
const API_BASE = 'http://localhost:8000';

function Skeleton({ className = '' }) {
    return <div className={`animate-pulse bg-white/10 rounded ${className}`} />;
}

function RegionSelector({ selected, onChange }) {
    return (
        <select
            value={selected}
            onChange={(e) => onChange(e.target.value)}
            className="bg-black border border-white/20 rounded px-3 py-2 text-sm focus:outline-none focus:border-white/50"
        >
            {REGIONS.map(r => (
                <option key={r} value={r}>{r}</option>
            ))}
        </select>
    );
}

function formatTime(isoString) {
    if (!isoString) return '--';
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit' });
}

function formatDateTime(isoString) {
    if (!isoString) return '--';
    const date = new Date(isoString);
    return date.toLocaleString('en-AU', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function PredictionCard({ prediction, currentPrice }) {
    if (!prediction) {
        return (
            <div className="p-6 bg-white/5 rounded-lg border border-white/10">
                <p className="text-white/40">No prediction available</p>
            </div>
        );
    }

    const diff = prediction.predicted_price - currentPrice;
    const diffPercent = currentPrice !== 0 ? (diff / currentPrice) * 100 : 0;
    const isUp = diff > 0;

    return (
        <div className="p-6 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg border border-white/20">
            <div className="flex justify-between items-start mb-4">
                <div>
                    <p className="text-white/40 text-xs uppercase tracking-widest mb-1">Predicted Price</p>
                    <p className="text-4xl font-light">${prediction.predicted_price.toFixed(2)}</p>
                </div>
                <div className={`px-3 py-1 rounded-full text-sm ${isUp ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                    {isUp ? '↑' : '↓'} {Math.abs(diffPercent).toFixed(1)}%
                </div>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                    <p className="text-white/40">Current Price</p>
                    <p className="text-white/80">${currentPrice?.toFixed(2) || '--'}</p>
                </div>
                <div>
                    <p className="text-white/40">Target Time</p>
                    <p className="text-white/80">{formatTime(prediction.target_time)}</p>
                </div>
            </div>
        </div>
    );
}

function ErrorTracker({ predictions }) {
    if (!predictions || predictions.length === 0) {
        return (
            <div className="p-4 bg-white/5 rounded-lg text-center text-white/40">
                <p>No completed predictions yet.</p>
                <p className="text-sm mt-1">Errors will appear after 15 minutes when actual prices arrive.</p>
            </div>
        );
    }

    return (
        <div className="space-y-2 max-h-80 overflow-y-auto">
            {predictions.slice().reverse().map((pred, i) => (
                <div key={i} className="p-3 bg-white/5 rounded-lg flex justify-between items-center">
                    <div className="flex-1">
                        <p className="text-sm text-white/60">{formatDateTime(pred.target_time)}</p>
                        <div className="flex gap-4 text-xs text-white/40 mt-1">
                            <span>Predicted: ${pred.predicted?.toFixed(2)}</span>
                            <span>Actual: ${pred.actual?.toFixed(2)}</span>
                        </div>
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${pred.error_percent < 5 ? 'bg-green-500/20 text-green-400' :
                            pred.error_percent < 15 ? 'bg-yellow-500/20 text-yellow-400' :
                                'bg-red-500/20 text-red-400'
                        }`}>
                        {pred.error_percent?.toFixed(1)}% error
                    </div>
                </div>
            ))}
        </div>
    );
}

function AccuracyMetrics({ accuracy }) {
    if (!accuracy || accuracy.total_predictions === 0) {
        return null;
    }

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-white/5 rounded-lg">
                <p className="text-white/40 text-xs uppercase tracking-widest mb-1">MAPE</p>
                <p className="text-2xl font-light">{accuracy.mape?.toFixed(2)}%</p>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
                <p className="text-white/40 text-xs uppercase tracking-widest mb-1">Median Error</p>
                <p className="text-2xl font-light">{accuracy.median_error?.toFixed(2)}%</p>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
                <p className="text-white/40 text-xs uppercase tracking-widest mb-1">Best</p>
                <p className="text-2xl font-light text-green-400">{accuracy.min_error?.toFixed(2)}%</p>
            </div>
            <div className="p-4 bg-white/5 rounded-lg">
                <p className="text-white/40 text-xs uppercase tracking-widest mb-1">Worst</p>
                <p className="text-2xl font-light text-red-400">{accuracy.max_error?.toFixed(2)}%</p>
            </div>
        </div>
    );
}

export default function PredictionsPage() {
    const [selectedRegion, setSelectedRegion] = useState('SA1');
    const [data, setData] = useState(null);
    const [accuracy, setAccuracy] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [serverOnline, setServerOnline] = useState(true);

    const fetchData = useCallback(async () => {
        try {
            const [predRes, accRes] = await Promise.all([
                fetch(`${API_BASE}/predictions/${selectedRegion}`),
                fetch(`${API_BASE}/accuracy/${selectedRegion}`)
            ]);

            if (!predRes.ok) {
                throw new Error('Prediction server not available');
            }

            const predData = await predRes.json();
            const accData = accRes.ok ? await accRes.json() : null;

            setData(predData);
            setAccuracy(accData);
            setError(null);
            setServerOnline(true);
        } catch (err) {
            setError(err.message);
            setServerOnline(false);
        }
        setLoading(false);
    }, [selectedRegion]);

    useEffect(() => {
        setLoading(true);
        fetchData();
    }, [selectedRegion, fetchData]);

    // Auto-refresh every 30 seconds
    useEffect(() => {
        const interval = setInterval(fetchData, 30 * 1000);
        return () => clearInterval(interval);
    }, [fetchData]);

    return (
        <div className="min-h-screen p-8 max-w-6xl mx-auto">
            <header className="mb-8 flex justify-between items-start">
                <div>
                    <div className="flex items-center gap-4 mb-2">
                        <Link href="/" className="text-white/40 hover:text-white/60 text-sm">
                            ← Dashboard
                        </Link>
                    </div>
                    <h1 className="text-3xl font-light tracking-tight">ML Predictions</h1>
                    <p className="text-white/40 text-sm mt-1">
                        15-minute ahead price forecasts with error tracking
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <RegionSelector selected={selectedRegion} onChange={setSelectedRegion} />
                    <div className={`px-3 py-1 rounded-full text-xs ${serverOnline ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}>
                        {serverOnline ? '● Server Online' : '○ Server Offline'}
                    </div>
                </div>
            </header>

            {/* Server Offline Warning */}
            {!serverOnline && (
                <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                    <p className="text-red-400 font-medium">Prediction Server Offline</p>
                    <p className="text-red-400/70 text-sm mt-1">
                        Start the server with: <code className="bg-black/50 px-2 py-1 rounded">python prediction_server.py</code>
                    </p>
                    <button
                        onClick={fetchData}
                        className="mt-3 px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded text-sm text-red-400"
                    >
                        Retry Connection
                    </button>
                </div>
            )}

            {/* Main Prediction Card */}
            {loading && !data ? (
                <Skeleton className="h-40 mb-8" />
            ) : (
                <div className="mb-8">
                    <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">
                        Next Price Prediction - {selectedRegion}
                    </h2>
                    <PredictionCard
                        prediction={data?.prediction}
                        currentPrice={data?.current_price}
                    />
                </div>
            )}

            {/* Accuracy Metrics */}
            {accuracy && accuracy.total_predictions > 0 && (
                <div className="mb-8">
                    <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">
                        Model Accuracy ({accuracy.total_predictions} predictions)
                    </h2>
                    <AccuracyMetrics accuracy={accuracy} />
                </div>
            )}

            {/* Two Column Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Pending Predictions */}
                <section>
                    <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">
                        Pending Predictions
                    </h2>
                    <div className="space-y-2">
                        {data?.pending_predictions?.length > 0 ? (
                            data.pending_predictions.map((pred, i) => (
                                <div key={i} className="p-3 bg-white/5 rounded-lg flex justify-between items-center">
                                    <div>
                                        <p className="text-sm text-white/60">Target: {formatTime(pred.target_time)}</p>
                                        <p className="text-xs text-white/40">Predicted: ${pred.predicted?.toFixed(2)}</p>
                                    </div>
                                    <div className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs">
                                        Waiting...
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="p-4 bg-white/5 rounded-lg text-center text-white/40 text-sm">
                                No pending predictions
                            </div>
                        )}
                    </div>
                </section>

                {/* Error History */}
                <section>
                    <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">
                        Prediction Errors
                    </h2>
                    <ErrorTracker predictions={data?.completed_predictions} />
                </section>
            </div>

            {/* Model Info */}
            <div className="mt-8 p-4 bg-white/5 rounded-lg text-xs text-white/40">
                <div className="flex flex-wrap gap-6">
                    <div>
                        <span className="text-white/60">Model:</span> {data?.model || 'XGBoost'}
                    </div>
                    <div>
                        <span className="text-white/60">Horizon:</span> 15 minutes
                    </div>
                    <div>
                        <span className="text-white/60">Last Trained:</span> {formatDateTime(data?.last_trained)}
                    </div>
                    <div>
                        <span className="text-white/60">Current Time:</span> {formatDateTime(data?.current_time)}
                    </div>
                </div>
            </div>

            <footer className="pt-8 mt-8 border-t border-white/5">
                <p className="text-white/20 text-xs">
                    Predictions generated by XGBoost model. Auto-refreshes every 30 seconds.
                    Errors calculated as |predicted - actual| / actual × 100%.
                </p>
            </footer>
        </div>
    );
}
