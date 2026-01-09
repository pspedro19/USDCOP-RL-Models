import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { NotificationProvider } from "@/components/ui/notification-manager";
import { ModelProvider } from "@/contexts/ModelContext";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "USDCOP Pro Trading Dashboard | Advanced Analytics & ML Predictions",
  description: "Professional real-time USDCOP trading dashboard with machine learning predictions, advanced charting, technical indicators, and comprehensive market analysis tools.",
  keywords: "USDCOP, trading, forex, machine learning, predictions, technical analysis, charts, dashboard, real-time, Colombian peso",
  authors: [{ name: "USDCOP Trading Team" }],
  robots: "index, follow",
  openGraph: {
    title: "USDCOP Pro Trading Dashboard",
    description: "Advanced trading analytics with ML-powered predictions for USDCOP forex pair",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "USDCOP Pro Trading Dashboard",
    description: "Professional trading dashboard with machine learning predictions",
  },
  icons: {
    icon: [
      { url: "/favicon.svg", type: "image/svg+xml" }
    ],
    apple: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Block Web3 wallet injections before page load
              (function() {
                if (typeof window !== 'undefined') {
                  // Try to define properties before wallets inject
                  const blockList = ['ethereum', 'web3', 'tronWeb', 'solana', 'phantom', 'tron'];
                  blockList.forEach(prop => {
                    try {
                      // Check if property already exists
                      const descriptor = Object.getOwnPropertyDescriptor(window, prop);
                      if (!descriptor) {
                        // Define it as undefined before wallet can inject
                        Object.defineProperty(window, prop, {
                          get: function() { return undefined; },
                          set: function() { return true; },
                          configurable: true
                        });
                      }
                    } catch (e) {
                      // Silently fail if property is protected
                    }
                  });
                  
                  // Block MetaMask message events
                  const originalAddEventListener = window.addEventListener;
                  window.addEventListener = function(type, listener, options) {
                    // Block MetaMask specific events
                    if (type === 'message') {
                      const wrappedListener = function(e) {
                        if (e.data && e.data.type && 
                            (e.data.type.includes('metamask') || 
                             e.data.type.includes('METAMASK') ||
                             e.data.type.includes('ethereum'))) {
                          e.stopImmediatePropagation();
                          return;
                        }
                        listener.call(this, e);
                      };
                      return originalAddEventListener.call(this, type, wrappedListener, options);
                    }
                    return originalAddEventListener.call(this, type, listener, options);
                  };
                }
              })();
            `,
          }}
        />
      </head>
      <body
        className={`${inter.className} antialiased bg-[#050816] text-slate-100`}
        suppressHydrationWarning
      >
        <ErrorBoundary
          level="page"
          maxRetries={1}
          showDetails={true}
        >
          <ModelProvider>
            <NotificationProvider maxNotifications={5} defaultDuration={6000}>
              {children}
            </NotificationProvider>
          </ModelProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}
