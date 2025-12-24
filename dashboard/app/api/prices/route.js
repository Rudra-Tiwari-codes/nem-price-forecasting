import { NextResponse } from 'next/server';

const NEMWEB_URL = 'https://www.nemweb.com.au/REPORTS/CURRENT/DispatchIS_Reports/';

async function getLatestZipLinks() {
    const response = await fetch(NEMWEB_URL, {
        headers: {
            'User-Agent': 'NEM-Analytics-Dashboard/1.0'
        }
    });
    if (!response.ok) throw new Error('Failed to fetch NEMWEB directory');

    const html = await response.text();

    // Extract ZIP file links from HTML directory listing
    const regex = /href="([^"]*DISPATCHIS[^"]*\.zip)"/gi;
    const matches = [...html.matchAll(regex)];

    const links = matches
        .map(m => {
            const href = m[1];
            if (href.startsWith('http')) return href;
            if (href.startsWith('/')) return 'https://www.nemweb.com.au' + href;
            return NEMWEB_URL + href.split('/').pop();
        })
        .slice(-5); // Get last 5 files

    return links;
}

async function extractPricesFromZip(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) return [];

        const buffer = await response.arrayBuffer();

        // Parse ZIP file manually (simple approach for single CSV)
        const bytes = new Uint8Array(buffer);

        // Find the CSV content in the ZIP
        // ZIP files have local file headers starting with PK\x03\x04
        let csvContent = '';

        // Simple extraction - find CSV data after the header
        const decoder = new TextDecoder('utf-8');
        const fullText = decoder.decode(bytes);

        // Find DISPATCH,PRICE data lines
        const lines = fullText.split('\n');
        const prices = [];

        for (const line of lines) {
            if (line.startsWith('D,DISPATCH,PRICE,')) {
                const parts = line.split(',');
                const settlementDate = parts[4]?.replace(/"/g, '');
                const regionId = parts[6];
                const rrp = parseFloat(parts[7]) || 0;

                if (regionId === 'SA1' && settlementDate) {
                    prices.push({
                        time: settlementDate,
                        region: regionId,
                        price: rrp
                    });
                }
            }
        }

        return prices;
    } catch (e) {
        console.error(`Failed to process ${url}:`, e);
        return [];
    }
}

export async function GET() {
    try {
        const zipLinks = await getLatestZipLinks();

        if (zipLinks.length === 0) {
            // Return demo data if no files found
            return returnDemoData('No dispatch files found on NEMWEB');
        }

        let allPrices = [];
        for (const url of zipLinks) {
            const prices = await extractPricesFromZip(url);
            allPrices.push(...prices);
        }

        // Sort by date and remove duplicates
        allPrices.sort((a, b) => new Date(a.time) - new Date(b.time));

        const uniquePrices = [];
        const seen = new Set();
        for (const p of allPrices) {
            if (!seen.has(p.time)) {
                seen.add(p.time);
                uniquePrices.push(p);
            }
        }

        const last100 = uniquePrices.slice(-100);

        if (last100.length === 0) {
            return returnDemoData('No SA1 price data extracted');
        }

        // Calculate EMA forecast
        const emaSpan = 12;
        let ema = last100[0]?.price || 0;
        const multiplier = 2 / (emaSpan + 1);

        const prices = last100.map((d, i) => {
            if (i > 0) {
                ema = (d.price - ema) * multiplier + ema;
            }

            let signal = 'hold';
            if (i > 0) {
                if (ema > d.price * 1.05) signal = 'buy';
                else if (ema < d.price * 0.95) signal = 'sell';
            }

            const date = new Date(d.time);
            const hours = String(date.getHours()).padStart(2, '0');
            const mins = String(date.getMinutes()).padStart(2, '0');

            return {
                time: `${hours}:${mins}`,
                fullDate: d.time,
                price: Math.round(d.price * 100) / 100,
                forecast: Math.round(ema * 100) / 100,
                signal
            };
        });

        const priceValues = last100.map(d => d.price);
        const stats = {
            current: priceValues[priceValues.length - 1],
            mean: Math.round(priceValues.reduce((a, b) => a + b, 0) / priceValues.length * 100) / 100,
            min: Math.min(...priceValues),
            max: Math.max(...priceValues),
            count: priceValues.length
        };

        return NextResponse.json({
            prices,
            stats,
            source: 'NEMWEB Live',
            filesProcessed: zipLinks.length,
            lastUpdated: new Date().toISOString()
        });

    } catch (error) {
        console.error('NEMWEB Scraper Error:', error);
        return returnDemoData(error.message);
    }
}

function returnDemoData(reason) {
    const now = Date.now();
    const prices = Array.from({ length: 50 }, (_, i) => {
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

    return NextResponse.json({
        prices,
        stats: {
            current: prices[prices.length - 1].price,
            mean: 105.5,
            min: 70,
            max: 150,
            count: 50
        },
        source: 'Demo Data',
        reason,
        lastUpdated: new Date().toISOString()
    });
}
