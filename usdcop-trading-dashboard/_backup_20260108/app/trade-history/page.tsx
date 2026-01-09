'use client';

import { TradesTable } from "@/components/trading/TradesTable";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { ArrowLeft, Download } from "lucide-react";

export default function TradeHistoryPage() {
    return (
        <main className="min-h-screen bg-[#050816] text-slate-200">
            {/* Header */}
            <header className="border-b border-slate-800 bg-[#0A0E27]/50 backdrop-blur-md sticky top-0 z-50">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/">
                            <Button variant="ghost" size="icon" className="text-slate-400 hover:text-white">
                                <ArrowLeft className="h-5 w-5" />
                            </Button>
                        </Link>
                        <h1 className="text-xl font-bold text-white">Trade History</h1>
                    </div>
                    <Button variant="outline" size="sm" className="border-slate-700 hover:bg-slate-800 text-slate-300">
                        <Download className="h-4 w-4 mr-2" /> Export CSV
                    </Button>
                </div>
            </header>

            <div className="container mx-auto px-4 py-8">
                <TradesTable />
            </div>
        </main>
    );
}
