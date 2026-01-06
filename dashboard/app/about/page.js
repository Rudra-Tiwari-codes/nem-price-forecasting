'use client';

import Link from 'next/link';

export default function About() {
    return (
        <div className="min-h-screen bg-black text-white font-sans selection:bg-purple-500/30">

            {/* Background Gradients */}
            <div className="fixed inset-0 z-0 pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-purple-900/20 rounded-full blur-[120px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-blue-900/20 rounded-full blur-[120px]" />
            </div>

            {/* Main Content Container */}
            <div className="relative z-10 pt-24 pb-16 sm:pt-32 sm:pb-24 px-6 max-w-4xl mx-auto">

                {/* Hero Section */}
                <header className="mb-32">
                    <h1 className="text-6xl sm:text-8xl font-medium tracking-tight mb-8 text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-white/50 font-[family-name:var(--font-space)]">
                        What is this?
                    </h1>
                    <p className="text-xl sm:text-3xl text-white/50 max-w-2xl leading-relaxed font-light">
                        This is a dashboard that watches Australia's electricity prices in real-time. It's built to answer one question: <span className="text-white">When is the best time to use energy?</span>
                    </p>
                </header>

                {/* Storytelling Section */}
                <div className="space-y-32">

                    <section className="grid md:grid-cols-[200px_1fr] gap-8 md:gap-16 border-t border-white/5 pt-12">
                        <h2 className="text-xs font-medium text-purple-400 uppercase tracking-[0.2em] font-[family-name:var(--font-space)]">
                            The Context
                        </h2>
                        <div className="space-y-8">
                            <h3 className="text-3xl sm:text-4xl text-white font-[family-name:var(--font-space)] leading-tight">
                                Electricity prices are a <span className="text-purple-400">rollercoaster</span>
                            </h3>
                            <div className="prose prose-invert prose-lg text-white/60 font-light leading-relaxed">
                                <p>
                                    Most people pay a flat rate for power, but the <em>actual</em> wholesale price of electricity changes every 5 minutes.
                                </p>
                                <p>
                                    Picture a hot summer evening: everyone gets home, turns on the AC, and starts cooking. Demand spikes, and prices go through the roof.
                                </p>
                                <p>
                                    Now picture a sunny Sunday morning: solar panels everywhere are pumping out energy, but factories are closed. There's too much power! Prices actually drop below zero, meaning <span className="text-white">generators pay the market to take their energy</span>.
                                </p>
                            </div>
                        </div>
                    </section>

                    <section className="grid md:grid-cols-[200px_1fr] gap-8 md:gap-16 border-t border-white/5 pt-12">
                        <h2 className="text-xs font-medium text-blue-400 uppercase tracking-[0.2em] font-[family-name:var(--font-space)]">
                            The Opportunity
                        </h2>
                        <div className="space-y-8">
                            <h3 className="text-3xl sm:text-4xl text-white font-[family-name:var(--font-space)] leading-tight">
                                Buy low, sell high
                            </h3>
                            <p className="text-white/60 leading-relaxed font-light text-lg">
                                This is where big batteries come in. It's the same concept as trading stocks, but with electrons.
                            </p>

                            <div className="grid gap-4 mt-4">
                                <div className="group p-6 rounded-2xl bg-white/5 border border-white/5 hover:border-white/10 hover:bg-white/[0.07] transition-all">
                                    <div className="flex gap-6 items-start">
                                        <span className="text-green-400 font-[family-name:var(--font-space)] text-2xl opacity-50 group-hover:opacity-100 transition-opacity">01</span>
                                        <div>
                                            <h4 className="text-white text-lg font-medium mb-1">Charge</h4>
                                            <p className="text-white/50 font-light">Load up the battery when the sun is shining and prices are negative.</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="group p-6 rounded-2xl bg-white/5 border border-white/5 hover:border-white/10 hover:bg-white/[0.07] transition-all">
                                    <div className="flex gap-6 items-start">
                                        <span className="text-blue-400 font-[family-name:var(--font-space)] text-2xl opacity-50 group-hover:opacity-100 transition-opacity">02</span>
                                        <div>
                                            <h4 className="text-white text-lg font-medium mb-1">Wait</h4>
                                            <p className="text-white/50 font-light">Hold the energy while the sun sets and grid demand begins to spike.</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="group p-6 rounded-2xl bg-white/5 border border-white/5 hover:border-white/10 hover:bg-white/[0.07] transition-all">
                                    <div className="flex gap-6 items-start">
                                        <span className="text-purple-400 font-[family-name:var(--font-space)] text-2xl opacity-50 group-hover:opacity-100 transition-opacity">03</span>
                                        <div>
                                            <h4 className="text-white text-lg font-medium mb-1">Discharge</h4>
                                            <p className="text-white/50 font-light">Sell that energy back to the grid for a massive profit.</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </section>

                    <section className="grid md:grid-cols-[200px_1fr] gap-8 md:gap-16 border-t border-white/5 pt-12">
                        <h2 className="text-xs font-medium text-green-400 uppercase tracking-[0.2em] font-[family-name:var(--font-space)]">
                            The Intelligence
                        </h2>
                        <div className="space-y-8">
                            <h3 className="text-3xl sm:text-4xl text-white font-[family-name:var(--font-space)] leading-tight">
                                Simulating future returns
                            </h3>
                            <p className="text-white/60 leading-relaxed font-light text-lg">
                                This dashboard connects directly to the market operator (AEMO) to get fresh data every minute. But it doesn't just show numbers.
                                It runs calculations to ask: <span className="text-white">"If I had a giant battery right now, what should I do?"</span>
                            </p>

                            <div className="grid sm:grid-cols-2 gap-6 mt-6">
                                <div className="p-8 rounded-3xl bg-gradient-to-br from-white/5 to-transparent border border-white/5 hover:border-white/10 transition-colors">
                                    <h4 className="text-xl text-white font-[family-name:var(--font-space)] mb-3">Naive Strategy</h4>
                                    <p className="text-sm text-white/50 font-light leading-relaxed">
                                        The simple approach. Set a fixed price target. "If it's under $0, buy. If it's over $300, sell." It's reliable but misses nuance.
                                    </p>
                                </div>
                                <div className="p-8 rounded-3xl bg-gradient-to-br from-white/5 to-transparent border border-white/5 hover:border-white/10 transition-colors">
                                    <h4 className="text-xl text-white font-[family-name:var(--font-space)] mb-3">Mean Reversion</h4>
                                    <p className="text-sm text-white/50 font-light leading-relaxed">
                                        The smart approach. Looks at the average price lately. If today is weirdly expensive compared to normal, sell now before it drops.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </section>

                    <section className="border-t border-white/5 pt-12">
                        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-8 text-[10px] text-white font-mono uppercase tracking-widest">
                            <div className="flex gap-8">
                                <div className="flex flex-col gap-1">
                                    <span className="text-white">Status</span>
                                    <span className="text-white">Online</span>
                                </div>
                                <div className="flex flex-col gap-1">
                                    <span className="text-white">Latency</span>
                                    <span>~45ms</span>
                                </div>
                                <div className="flex flex-col gap-1">
                                    <span className="text-white">Source</span>
                                    <span>AEMO NEMWEB</span>
                                </div>
                            </div>
                            <div>
                                NEM Analytics Engine v1.2.0
                            </div>
                        </div>
                    </section>

                </div>

                {/* CTA */}
                <section className="mt-32 pt-12 border-t border-white/5 text-center">
                    <Link
                        href="/"
                        className="group relative inline-flex items-center gap-3 px-8 py-4 bg-white/5 border border-white/10 rounded-full overflow-hidden hover:border-white/20 hover:bg-white/10 transition-all duration-300 hover:scale-105 hover:shadow-[0_0_40px_-10px_rgba(168,85,247,0.5)]"
                    >
                        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-blue-600/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                        <span className="relative z-10 text-lg font-[family-name:var(--font-space)] text-white tracking-wide">Check Live Prices</span>
                        <svg className="w-5 h-5 relative z-10 text-purple-400 transition-transform group-hover:translate-x-1 group-hover:text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                        </svg>
                    </Link>
                </section>

            </div>
        </div>
    );
}
