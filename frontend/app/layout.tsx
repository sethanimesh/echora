import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {
  title: 'Echora — Your voice, your words',
  description: 'A personal space to express, review, and speak your message.',
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
