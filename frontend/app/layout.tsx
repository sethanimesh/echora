import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "http://localhost:3000"),
  title: "Echora · Say it your way",
  description: "Local-first communication support for difficult-to-understand speech.",
  openGraph: {
    title: "Echora · Say it your way",
    description: "Literal speech. Honest choices. Your confirmation.",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "Echora — Say it your way" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Echora · Say it your way",
    description: "Literal speech. Honest choices. Your confirmation.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
