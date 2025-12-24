'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart } from 'recharts';

export default function Home() {
  const [priceData, setPriceData] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    fetchData();
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const res = await fetch('/api/prices');
      const data = await res.json();
      setPriceData(data.prices || []);
      setStats(data.stats || null);
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setLoading(false);
    }
  };

  if (!mounted || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-4 h-4 border border-white/30 border-t-white rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="mb-12">
        <h1 className="text-3xl font-light tracking-tight">NEM Analytics</h1>
        <p className="text-white/40 text-sm mt-1">Real-time electricity market data</p>
      </header>

      {stats && (
        <div className="grid grid-cols-4 gap-6 mb-12">
          <Stat label="Current" value={stats.current?.toFixed(2)} />
          <Stat label="Average" value={stats.mean?.toFixed(2)} />
          <Stat label="High" value={stats.max?.toFixed(2)} />
          <Stat label="Low" value={stats.min?.toFixed(2)} />
        </div>
      )}

      <section className="mb-12">
        <h2 className="text-sm text-white/40 uppercase tracking-widest mb-6">Price & Forecast</h2>
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
                name="Forecast"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="grid grid-cols-2 gap-12">
        <section>
          <h2 className="text-sm text-white/40 uppercase tracking-widest mb-6">Signals</h2>
          <div className="space-y-2">
            {priceData.slice(-8).reverse().map((d, i) => (
              <div key={i} className="flex justify-between items-center py-2 border-b border-white/5">
                <span className="text-white/50 text-sm">{d.time}</span>
                <span className={`text-xs uppercase tracking-wider ${d.signal === 'buy' ? 'text-white' :
                    d.signal === 'sell' ? 'text-white/30' :
                      'text-white/20'
                  }`}>
                  {d.signal || 'hold'}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h2 className="text-sm text-white/40 uppercase tracking-widest mb-6">Analysis</h2>
          <div className="space-y-3">
            <ChartLink title="Distribution" href="/charts/eda_price_distribution.png" />
            <ChartLink title="Volatility" href="/charts/eda_volatility.png" />
            <ChartLink title="Patterns" href="/charts/eda_temporal_patterns.png" />
            <ChartLink title="Outliers" href="/charts/eda_outliers.png" />
          </div>
        </section>
      </div>

      <footer className="mt-20 pt-8 border-t border-white/5">
        <p className="text-white/20 text-xs">Data from AEMO. Updates every 5 minutes.</p>
      </footer>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div>
      <p className="text-white/40 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className="text-2xl font-light">${value}</p>
    </div>
  );
}

function ChartLink({ title, href }) {
  return (
    <a
      href={href}
      target="_blank"
      className="block py-3 px-4 border border-white/10 rounded hover:border-white/30 transition-colors"
    >
      <span className="text-sm">{title}</span>
    </a>
  );
}
