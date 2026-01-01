import { NextResponse } from 'next/server';
import JSZip from 'jszip';

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

async function extractPricesFromZip(url, region = 'SA1') {
    try {
        const response = await fetch(url);
        if (!response.ok) return [];

        const buffer = await response.arrayBuffer();

        // Use JSZip to properly decompress the ZIP file
        const zip = await JSZip.loadAsync(buffer);

        // Find the CSV file inside the ZIP
        const csvFileName = Object.keys(zip.files).find(name =>
            name.toUpperCase().endsWith('.CSV')
        );

        if (!csvFileName) {
            console.log(`No CSV found in ${url}`);
            return [];
        }

        // Extract and read the CSV content
        const csvContent = await zip.files[csvFileName].async('string');
        const lines = csvContent.split('\n');
        const prices = [];

        // Find column indices from header row
        let settlementDateIdx = -1;
        let regionIdIdx = -1;
        let rrpIdx = -1;

        for (const line of lines) {
            // Header row starts with I,DISPATCH,PRICE
            if (line.startsWith('I,DISPATCH,PRICE,')) {
                const headerParts = line.split(',');
                for (let i = 0; i < headerParts.length; i++) {
                    const col = headerParts[i].replace(/"/g, '').trim().toUpperCase();
                    if (col === 'SETTLEMENTDATE') settlementDateIdx = i;
                    if (col === 'REGIONID') regionIdIdx = i;
                    if (col === 'RRP') rrpIdx = i;
                }
            }
            // Data row starts with D,DISPATCH,PRICE
            if (line.startsWith('D,DISPATCH,PRICE,') && settlementDateIdx >= 0 && regionIdIdx >= 0 && rrpIdx >= 0) {
                const parts = line.split(',');
                const settlementDate = parts[settlementDateIdx]?.replace(/"/g, '');
                const regionId = parts[regionIdIdx]?.replace(/"/g, '');
                const rrp = parseFloat(parts[rrpIdx]) || 0;

                if (regionId === region && settlementDate) {
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

export async function GET(request) {
    try {
        // Get region from query parameter, default to SA1
        const { searchParams } = new URL(request.url);
        const selectedRegion = searchParams.get('region') || 'SA1';
        const validRegions = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1'];

        if (!validRegions.includes(selectedRegion)) {
            return NextResponse.json({
                error: `Invalid region. Valid regions: ${validRegions.join(', ')}`,
                prices: [],
                stats: null
            }, { status: 400 });
        }

        // First, try to use pre-generated simulation data (has 100 data points)
        try {
            const simUrl = `https://raw.githubusercontent.com/Rudra-Tiwari-codes/nem-price-forecasting/main/dashboard/public/simulation_${selectedRegion}.json`;
            const simResponse = await fetch(simUrl, { next: { revalidate: 300 } });

            if (simResponse.ok) {
                const simData = await simResponse.json();
                if (simData.prices && simData.prices.length > 0) {
                    return NextResponse.json({
                        prices: simData.prices,
                        stats: simData.stats,
                        region: selectedRegion,
                        source: 'GitHub Simulation Data',
                        filesProcessed: 0,
                        lastUpdated: simData.lastUpdated || new Date().toISOString()
                    });
                }
            }
        } catch (simErr) {
            console.log('Simulation data not available, falling back to NEMWEB:', simErr.message);
        }

        // Fallback: Try live NEMWEB scraping
        const zipLinks = await getLatestZipLinks();

        if (zipLinks.length === 0) {
            return returnDataError('No dispatch files found on NEMWEB');
        }

        let allPrices = [];
        for (const url of zipLinks) {
            const prices = await extractPricesFromZip(url, selectedRegion);
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
            return returnDataError('No ' + selectedRegion + ' price data extracted from ' + zipLinks.length + ' files');
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
            region: selectedRegion,
            source: 'NEMWEB Live',
            filesProcessed: zipLinks.length,
            lastUpdated: new Date().toISOString()
        });

    } catch (error) {
        console.error('NEMWEB Scraper Error:', error);
        return returnDataError(error.message);
    }
}

function returnDataError(reason) {
    // Return proper HTTP error instead of masking failures with fake data
    return NextResponse.json({
        error: 'Unable to fetch real-time price data',
        reason: reason,
        suggestion: 'Please try again later or check if AEMO NEMWEB is accessible',
        timestamp: new Date().toISOString()
    }, { status: 503 });
}

