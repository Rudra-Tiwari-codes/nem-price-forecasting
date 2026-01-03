import { NextResponse } from 'next/server';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

// Cache configuration
const CACHE_MAX_AGE = 300; // 5 minutes
const CACHE_STALE_WHILE_REVALIDATE = 60;

function addCacheHeaders(response) {
    response.headers.set('Cache-Control', `public, s-maxage=${CACHE_MAX_AGE}, stale-while-revalidate=${CACHE_STALE_WHILE_REVALIDATE}`);
    return response;
}

export async function GET(request) {
    try {
        // Get region from query parameter, default to SA1
        const { searchParams } = new URL(request.url);
        const region = searchParams.get('region') || 'SA1';
        const validRegions = ['SA1', 'NSW1', 'VIC1', 'QLD1', 'TAS1'];

        if (!validRegions.includes(region)) {
            return NextResponse.json({
                error: `Invalid region. Valid regions: ${validRegions.join(', ')}`,
                source: 'error'
            }, { status: 400 });
        }

        // In development, try local files first (more up-to-date than GitHub)
        const isDev = process.env.NODE_ENV === 'development';
        const githubBaseUrl = process.env.NEXT_PUBLIC_GITHUB_DATA_URL || 'https://raw.githubusercontent.com/Rudra-Tiwari-codes/nem-price-forecasting/main/dashboard/public';

        if (isDev) {
            try {
                const publicPath = join(process.cwd(), 'public', `simulation_${region}.json`);
                if (existsSync(publicPath)) {
                    const fileContent = readFileSync(publicPath, 'utf-8');
                    const data = JSON.parse(fileContent);
                    return addCacheHeaders(NextResponse.json({
                        ...data,
                        source: 'Local Simulation Data'
                    }));
                }
            } catch (localErr) {
                console.log('Local file read failed, falling back to GitHub:', localErr.message);
            }
        }

        // Fallback: Fetch simulation data from GitHub (works on Vercel serverless)
        const githubUrl = `${githubBaseUrl}/simulation_${region}.json`;
        const response = await fetch(githubUrl, { next: { revalidate: 60 } });

        if (response.ok) {
            const data = await response.json();
            return addCacheHeaders(NextResponse.json({
                ...data,
                source: 'GitHub Simulation Data'
            }));
        }

        // Fallback to SA1 if specific region not found
        if (region !== 'SA1') {
            const sa1Url = `${githubBaseUrl}/simulation_SA1.json`;
            const sa1Response = await fetch(sa1Url, { next: { revalidate: 60 } });

            if (sa1Response.ok) {
                const data = await sa1Response.json();
                return addCacheHeaders(NextResponse.json({
                    ...data,
                    source: 'GitHub (SA1 fallback)'
                }));
            }
        }

        // If no simulation results available, return error
        return NextResponse.json({
            error: 'Simulation results not available. GitHub Actions may still be running.',
            source: 'none'
        }, { status: 404 });

    } catch (error) {
        console.error('Error fetching simulation results:', error);
        return NextResponse.json({
            error: error.message,
            source: 'error'
        }, { status: 500 });
    }
}

