import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
    try {
        const dataPath = path.join(process.cwd(), '..', 'data', 'combined_dispatch_prices.csv');

        let prices = [];
        let stats = null;

        if (fs.existsSync(dataPath)) {
            const content = fs.readFileSync(dataPath, 'utf-8');
            const lines = content.trim().split('\n');
            const headers = lines[0].split(',');

            const dateIdx = headers.findIndex(h => h.includes('SETTLEMENTDATE') || h.includes('DATE'));
            const priceIdx = headers.findIndex(h => h.includes('RRP') || h.includes('PRICE'));

            const data = lines.slice(1).map(line => {
                const parts = line.split(',');
                return {
                    date: parts[dateIdx],
                    price: parseFloat(parts[priceIdx]) || 0
                };
            }).filter(d => !isNaN(d.price));

            const last100 = data.slice(-100);

            const emaSpan = 12;
            let ema = last100[0]?.price || 0;
            const multiplier = 2 / (emaSpan + 1);

            prices = last100.map((d, i) => {
                if (i > 0) {
                    ema = (d.price - ema) * multiplier + ema;
                }

                let signal = 'hold';
                if (i > 0) {
                    if (ema > d.price * 1.05) signal = 'buy';
                    else if (ema < d.price * 0.95) signal = 'sell';
                }

                const date = new Date(d.date);
                const hours = String(date.getHours()).padStart(2, '0');
                const mins = String(date.getMinutes()).padStart(2, '0');
                const time = `${hours}:${mins}`;

                return {
                    time,
                    fullDate: d.date,
                    price: Math.round(d.price * 100) / 100,
                    forecast: Math.round(ema * 100) / 100,
                    signal
                };
            });

            const allPrices = data.map(d => d.price);
            stats = {
                current: allPrices[allPrices.length - 1],
                mean: allPrices.reduce((a, b) => a + b, 0) / allPrices.length,
                min: Math.min(...allPrices),
                max: Math.max(...allPrices),
                count: allPrices.length
            };
        } else {
            const now = Date.now();
            prices = Array.from({ length: 50 }, (_, i) => {
                const timestamp = now - (50 - i) * 5 * 60 * 1000;
                const date = new Date(timestamp);
                const hours = String(date.getHours()).padStart(2, '0');
                const mins = String(date.getMinutes()).padStart(2, '0');
                const basePrice = 100 + Math.sin(i / 5) * 30 + (i % 10);
                return {
                    time: `${hours}:${mins}`,
                    price: Math.round(basePrice * 100) / 100,
                    forecast: Math.round((basePrice + 5 - (i % 10)) * 100) / 100,
                    signal: i % 5 === 0 ? 'buy' : i % 7 === 0 ? 'sell' : 'hold'
                };
            });

            stats = {
                current: prices[prices.length - 1].price,
                mean: 105.5,
                min: 70,
                max: 150,
                count: 50
            };
        }

        return NextResponse.json({ prices, stats });
    } catch (error) {
        console.error('API Error:', error);
        return NextResponse.json({ error: 'Failed to fetch data', prices: [], stats: null }, { status: 500 });
    }
}
