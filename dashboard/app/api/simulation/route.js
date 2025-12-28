import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET(request) {
    try {
        // Get region from query parameter, default to SA1
        const { searchParams } = new URL(request.url);
        const region = searchParams.get('region') || 'SA1';

        // Try region-specific file first (matches main.py output)
        const regionPath = path.join(process.cwd(), 'public', `simulation_${region}.json`);

        if (fs.existsSync(regionPath)) {
            const data = JSON.parse(fs.readFileSync(regionPath, 'utf8'));
            return NextResponse.json({
                ...data,
                source: 'Python Simulation'
            });
        }

        // Fallback to SA1 if specific region not found
        const sa1Path = path.join(process.cwd(), 'public', 'simulation_SA1.json');
        if (fs.existsSync(sa1Path)) {
            const data = JSON.parse(fs.readFileSync(sa1Path, 'utf8'));
            return NextResponse.json({
                ...data,
                source: 'Python Simulation (SA1 fallback)'
            });
        }

        // Fallback: try to fetch from GitHub if local file doesn't exist
        const githubUrl = `https://raw.githubusercontent.com/Rudra-Tiwari-codes/nem-price-forecasting/main/dashboard/public/simulation_${region}.json`;
        const response = await fetch(githubUrl, { next: { revalidate: 60 } });

        if (response.ok) {
            const data = await response.json();
            return NextResponse.json({
                ...data,
                source: 'GitHub (Python Simulation)'
            });
        }

        // If no simulation results available, return error
        return NextResponse.json({
            error: 'Simulation results not available. Run python main.py first.',
            source: 'none'
        }, { status: 404 });

    } catch (error) {
        console.error('Error reading simulation results:', error);
        return NextResponse.json({
            error: error.message,
            source: 'error'
        }, { status: 500 });
    }
}
