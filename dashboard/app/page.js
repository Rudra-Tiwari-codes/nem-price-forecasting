'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, ComposedChart, BarChart, Bar, Line } from 'recharts';

const REGIONS = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1'];

// Skeleton loader component
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

function Stat({ label, value, highlight }) {
  return (
    <div className={`p-4 rounded-lg ${highlight ? 'bg-white/5 border border-white/10' : ''}`}>
      <p className="text-white/40 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-light ${highlight ? 'text-white' : 'text-white/80'}`}>
        ${typeof value === 'number' ? value.toFixed(2) : value}
      </p>
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRegion, setSelectedRegion] = useState('SA1');

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);

      const [pricesRes, simRes] = await Promise.all([
        fetch(`/api/prices?region=${selectedRegion}`, { signal: controller.signal }),
        fetch(`/api/simulation?region=${selectedRegion}`, { signal: controller.signal })
      ]);

      clearTimeout(timeout);

      let pricesData = null;
      let simData = null;

      if (pricesRes.ok) pricesData = await pricesRes.json();
      if (simRes.ok) simData = await simRes.json();

      const prices = pricesData?.prices || simData?.prices || [];
      const stats = pricesData?.stats || simData?.stats;

      if (prices.length > 0) {
        setData({
          ...simData,
          prices,
          stats,
          region: selectedRegion,
          source: pricesData?.source || simData?.source || 'API',
          lastUpdated: pricesData?.lastUpdated || simData?.lastUpdated || new Date().toISOString()
        });
        setError(null);
      } else {
        setError(`No data for ${selectedRegion}`);
      }
    } catch (err) {
      setError(err.name === 'AbortError' ? 'Timeout' : 'Failed to load');
    }
    setLoading(false);
  }, [selectedRegion]);

  useEffect(() => {
    fetchData();
  }, [selectedRegion, fetchData]);

  useEffect(() => {
    const interval = setInterval(fetchData, 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Always render the UI immediately - no blocking loading screen
  return (
    <div className="min-h-screen p-8 max-w-6xl mx-auto">
      <header className="mb-8 flex justify-between items-start">
        <div>
          <div className="flex items-center gap-4 mb-1">
            <h1 className="text-3xl font-light tracking-tight">NEM Analytics</h1>
            <Link href="/predictions" className="text-sm text-white/50 hover:text-white/80 border border-white/20 hover:border-white/40 px-3 py-1 rounded transition-colors">
              Predictions →
            </Link>
          </div>
          <p className="text-white/40 text-sm">
            {loading ? 'Loading...' : (data?.source || 'API')} | {selectedRegion}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <RegionSelector selected={selectedRegion} onChange={setSelectedRegion} />
          <div className="text-right">
            <p className="text-xs text-white/30">Last Updated</p>
            <p className="text-sm text-white/60">
              {data?.lastUpdated ? new Date(data.lastUpdated).toLocaleString() : '--'}
            </p>
          </div>
        </div>
      </header>

      {/* Error Banner (non-blocking) */}
      {error && !loading && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={fetchData} className="text-xs underline">Retry</button>
        </div>
      )}

      {/* Stats Grid - show skeleton when loading */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {loading && !data ? (
          <>
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
          </>
        ) : data?.stats ? (
          <>
            <Stat label="Current Price" value={data.stats.current} highlight />
            <Stat label="Average" value={data.stats.mean} />
            <Stat label="High" value={data.stats.max} />
            <Stat label="Low" value={data.stats.min} />
          </>
        ) : null}
      </div>

      {/* Best Strategy Banner */}
      {data?.bestStrategy && (
        <div className="mb-8 p-4 bg-white/5 rounded-lg border border-white/10">
          <div className="flex flex-wrap justify-between items-center gap-4">
            <div>
              <p className="text-white/40 text-xs uppercase tracking-widest">Best Strategy</p>
              <p className="text-xl font-light">{data.bestStrategy}</p>
            </div>
            <div className="text-right">
              <p className="text-white/40 text-xs uppercase tracking-widest">Total Profit</p>
              <p className="text-xl font-light text-green-400">${data.bestProfit?.toLocaleString()}</p>
            </div>
            <div className="text-right">
              <p className="text-white/40 text-xs uppercase tracking-widest">Annualized</p>
              <p className="text-xl font-light">
                {data.annualizedProfit != null
                  ? `$${data.annualizedProfit.toLocaleString()}/yr`
                  : <span className="text-white/40 text-sm">N/A (limited data)</span>
                }
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Price Chart - show skeleton when loading */}
      <section className="mb-8">
        <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">
          Price History - {selectedRegion} (Last ~8 Hours) {data?.prices?.length ? `- ${data.prices.length} points` : ''}
        </h2>
        <div className="h-64 border border-white/10 rounded-lg p-4">
          {loading && !data ? (
            <Skeleton className="w-full h-full" />
          ) : data?.prices?.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data.prices}>
                <CartesianGrid stroke="#222" vertical={false} />
                <XAxis
                  dataKey="time"
                  stroke="#444"
                  fontSize={10}
                  tickLine={false}
                  axisLine={false}
                  interval={Math.floor((data.prices?.length || 1) / 12)}
                />
                <YAxis stroke="#444" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `$${v}`} />
                <Tooltip
                  contentStyle={{ background: '#111', border: '1px solid #333', borderRadius: 4, fontSize: 11 }}
                  formatter={(value) => [`$${value}`, 'Price']}
                  labelFormatter={(label, payload) => payload?.[0]?.payload?.fullDate || label}
                />
                <Area type="monotone" dataKey="price" stroke="none" fill="#fff" fillOpacity={0.05} />
                <Line type="monotone" dataKey="price" stroke="#fff" strokeWidth={1.5} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-white/30">No chart data</div>
          )}
        </div>
      </section>

      {/* Strategy Comparison */}
      {data?.strategies?.length > 0 && (
        <section className="mb-8">
          <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">Strategy Comparison</h2>
          <div className="h-48 border border-white/10 rounded-lg p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.strategies} layout="vertical">
                <CartesianGrid stroke="#222" horizontal={false} />
                <XAxis type="number" stroke="#444" fontSize={10} tickFormatter={(v) => `$${v.toLocaleString()}`} />
                <YAxis type="category" dataKey="name" stroke="#444" fontSize={10} width={120} />
                <Tooltip
                  contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 11 }}
                  formatter={(value) => [`$${value.toLocaleString()}`, 'Profit']}
                />
                <Bar dataKey="profit" fill="#4ade80" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Trading Signals */}
        <section>
          <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">Trading Signals</h2>
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {(data?.signals || data?.prices?.slice(-20) || []).slice().reverse().map((s, i) => (
              <div key={i} className="flex justify-between items-center py-2 px-3 bg-white/5 rounded">
                <span className="text-white/50 text-sm">{s.time}</span>
                <div className="flex items-center gap-3">
                  <span className="text-white/40 text-sm">${s.price?.toFixed(2)}</span>
                  <span className={`text-xs uppercase tracking-wider px-2 py-1 rounded ${s.signal === 'buy' ? 'bg-green-500/20 text-green-400' :
                    s.signal === 'sell' ? 'bg-red-500/20 text-red-400' :
                      'bg-white/10 text-white/30'
                    }`}>
                    {s.signal || 'hold'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Data Info */}
        <section>
          <h2 className="text-sm text-white/40 uppercase tracking-widest mb-4">Data Info</h2>
          <div className="space-y-3">
            <div className="p-3 bg-white/5 rounded text-xs text-white/40">
              <p><strong>Source:</strong> {data?.source || 'API'}</p>
              <p><strong>Region:</strong> {data?.region || selectedRegion}</p>
              <p><strong>Data Points:</strong> {data?.prices?.length || 0}</p>
            </div>
            {data?.dataRange && (
              <div className="p-3 bg-white/5 rounded text-xs text-white/40">
                <p>Data Range: {data.dataRange.days} day(s)</p>
                <p className="truncate">From: {data.dataRange.start}</p>
                <p className="truncate">To: {data.dataRange.end}</p>
              </div>
            )}
          </div>
        </section>
      </div>

      <footer className="pt-8 border-t border-white/5">
        <p className="text-white/20 text-xs">
          Live data from AEMO NEMWEB via API. Simulation results from Python backend. Auto-refreshes every 60 seconds.
        </p>
      </footer>
    </div>
  );
}
