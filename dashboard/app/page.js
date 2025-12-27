'use client';

import { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from 'recharts';

const REGIONS = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1'];
const GITHUB_RAW_URL = 'https://raw.githubusercontent.com/Rudra-Tiwari-codes/nem-price-forecasting/main';

export default function Home() {
  const [priceData, setPriceData] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);
  const [region, setRegion] = useState('SA1');
  const [lastUpdated, setLastUpdated] = useState(null);
  const [source, setSource] = useState('');

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`/api/prices?region=${region}`);
      const data = await res.json();
      setPriceData(data.prices || []);
      setStats(data.stats || null);
      setLastUpdated(data.lastUpdated ? new Date(data.lastUpdated) : new Date());
      setSource(data.source || 'NEMWEB');
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setLoading(false);
    }
  }, [region]);

  useEffect(() => {
    setMounted(true);
    fetchData();
    // Update every 30 seconds for more real-time feel
    const interval = setInterval(fetchData, 30 * 1000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (!mounted || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-4 h-4 border border-white/30 border-t-white rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-light tracking-tight">NEM Analytics</h1>
          <p className="text-white/40 text-sm mt-1">Real-time electricity market data</p>
        </div>
        <div className="flex items-center gap-4">
          <select
            value={region}
            onChange={(e) => { setRegion(e.target.value); setLoading(true); }}
            className="bg-black border border-white/20 rounded px-3 py-2 text-sm focus:outline-none focus:border-white/50"
          >
            {REGIONS.map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
          <button
            onClick={() => { setLoading(true); fetchData(); }}
            className="text-white/50 hover:text-white text-sm px-3 py-2 border border-white/20 rounded hover:border-white/40 transition-colors"
          >
            Refresh
          </button>
        </div>
      </header>

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          <Stat label="Current" value={stats.current?.toFixed(2)} highlight />
          <Stat label="Average" value={stats.mean?.toFixed(2)} />
          <Stat label="High" value={stats.max?.toFixed(2)} />
          <Stat label="Low" value={stats.min?.toFixed(2)} />
        </div>
      )}

      <div className="flex justify-between items-center mb-4">
        <h2 className="text-sm text-white/40 uppercase tracking-widest">Price & Forecast</h2>
        <div className="text-xs text-white/30">
          {source} | {lastUpdated?.toLocaleTimeString()}
        </div>
      </div>

      <section className="mb-10">
        <div className="h-80 border border-white/10 rounded-lg p-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={priceData}>
              <CartesianGrid stroke="#222" strokeDasharray="0" vertical={false} />
              <XAxis
                dataKey="time"
                stroke="#444"
                fontSize={11}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="#444"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `$${v}`}
              />
              <Tooltip
                contentStyle={{
                  background: '#111',
                  border: '1px solid #333',
                  borderRadius: 4,
                  fontSize: 12
                }}
                labelStyle={{ color: '#666' }}
              />
              <Area
                type="monotone"
                dataKey="price"
                stroke="none"
                fill="#fff"
                fillOpacity={0.05}
              />
              <Line
                type="monotone"
                dataKey="price"
                stroke="#fff"
                strokeWidth={1.5}
                dot={false}
                name="Price"
              />
              <Line
                type="monotone"
                dataKey="forecast"
                stroke="#666"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
                name="EMA Forecast"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mb-10">
        <section>
          <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">Trading Signals</h2>
          <div className="space-y-1">
            {priceData.slice(-10).reverse().map((d, i) => (
              <div key={i} className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-white/50 text-sm">{d.time}</span>
                <div className="flex items-center gap-3">
                  <span className="text-white/40 text-sm">${d.price?.toFixed(2)}</span>
                  <span className={`text-xs uppercase tracking-wider px-2 py-1 rounded ${d.signal === 'buy' ? 'bg-green-500/20 text-green-400' :
                      d.signal === 'sell' ? 'bg-red-500/20 text-red-400' :
                        'bg-white/5 text-white/30'
                    }`}>
                    {d.signal || 'hold'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">Analysis Charts</h2>
          <div className="grid grid-cols-2 gap-3">
            <ChartLink title="Price Distribution" href={`${GITHUB_RAW_URL}/charts/price_distribution.png`} />
            <ChartLink title="Strategy Comparison" href={`${GITHUB_RAW_URL}/charts/strategy_comparison.png`} />
            <ChartLink title="Volatility Analysis" href={`${GITHUB_RAW_URL}/charts/eda_volatility.png`} />
            <ChartLink title="Temporal Patterns" href={`${GITHUB_RAW_URL}/charts/eda_temporal_patterns.png`} />
          </div>
        </section>
      </div>

      <footer className="pt-8 border-t border-white/5">
        <p className="text-white/20 text-xs">
          Data from AEMO NEMWEB. Auto-refreshes every 30 seconds. Region: {region}
        </p>
      </footer>
    </div>
  );
}

function Stat({ label, value, highlight }) {
  return (
    <div className={`p-4 rounded-lg ${highlight ? 'bg-white/5' : ''}`}>
      <p className="text-white/40 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-light ${highlight ? 'text-white' : 'text-white/80'}`}>${value}</p>
    </div>
  );
}

function ChartLink({ title, href }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="block py-3 px-4 border border-white/10 rounded hover:border-white/30 transition-colors text-center"
    >
      <span className="text-sm text-white/70">{title}</span>
    </a>
  );
}
