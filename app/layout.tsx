import type { Metadata } from 'next';
import './globals.css';
import Sidebar from '@/components/Sidebar';

export const metadata: Metadata = {
  title: 'Cortex',
  description: 'Personal biometrics analytics',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="flex h-screen overflow-hidden" style={{ background: '#0D1117' }}>
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-[1400px] mx-auto px-6 py-5">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
