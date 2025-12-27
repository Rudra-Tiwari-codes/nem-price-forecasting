import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
    try {
        // Try to read the simulation results JSON file
        const resultsPath = path.join(process.cwd(), 'public', 'simulation_results.json');

        if (fs.existsSync(resultsPath)) {
            const data = JSON.parse(fs.readFileSync(resultsPath, 'utf8'));
            return NextResponse.json({
                ...data,
                source: 'Python Simulation'
            });
        }

        // Fallback: try to fetch from GitHub if local file doesn't exist
        const githubUrl = 'https://raw.githubusercontent.com/Rudra-Tiwari-codes/nem-price-forecasting/main/dashboard/public/simulation_results.json';
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
