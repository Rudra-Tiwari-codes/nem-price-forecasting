import { Inter } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter"
});

export const metadata = {
  title: "NEM Analytics",
  description: "Electricity market price forecasting",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans antialiased bg-black text-white">
        <nav className="border-b border-white/10 px-8 py-3">
          <div className="max-w-6xl mx-auto flex gap-6 text-sm">
            <Link href="/" className="text-white/60 hover:text-white transition-colors">
              Dashboard
            </Link>
          </div>
        </nav>
        {children}
      </body>
    </html>
  );
}

