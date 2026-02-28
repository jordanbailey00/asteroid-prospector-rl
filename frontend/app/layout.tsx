import type { Metadata } from "next";
import Link from "next/link";
import { Barlow_Condensed, JetBrains_Mono, Space_Grotesk } from "next/font/google";

import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-space-grotesk",
});

const barlowCondensed = Barlow_Condensed({
  subsets: ["latin"],
  weight: ["400", "600"],
  variable: "--font-barlow-condensed",
});

const jetBrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "600"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: "Asteroid Prospector Console",
  description:
    "Replay, live metrics, and human pilot controls for Asteroid Prospecting RL runs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${spaceGrotesk.variable} ${barlowCondensed.variable} ${jetBrainsMono.variable}`}
      >
        <div className="app-bg" />
        <div className="app-shell">
          <header className="topbar">
            <div className="brand-block">
              <span className="brand-kicker">Asteroid Belt Prospector</span>
              <h1>Mission Control Interface</h1>
            </div>
            <nav className="nav-strip">
              <Link href="/">Replay</Link>
              <Link href="/play">Play</Link>
              <Link href="/analytics">Analytics</Link>
            </nav>
          </header>
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
