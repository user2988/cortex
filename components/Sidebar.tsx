'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV = [
  { href: '/',              label: 'Dashboard' },
  { href: '/experiments',   label: 'Experiments' },
  { href: '/explorer',      label: 'Explorer' },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <aside style={{
      width: 180, minWidth: 180, background: '#161B22',
      borderRight: '1px solid #21262D', display: 'flex', flexDirection: 'column',
      padding: '20px 0',
    }}>
      <div style={{ padding: '0 16px 16px' }}>
        <div style={{ fontFamily: 'Inter', fontSize: 16, fontWeight: 600, color: '#E6EDF3', letterSpacing: '-0.02em' }}>Cortex</div>
        <div style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 9, color: '#484F58', marginTop: 3, letterSpacing: '0.06em' }}>PERSONAL ANALYTICS</div>
      </div>
      <div style={{ height: 1, background: '#21262D', margin: '0 16px 12px' }} />
      <nav>
        {NAV.map(({ href, label }) => {
          const active = href === '/' ? path === '/' : path.startsWith(href);
          return (
            <Link key={href} href={href} style={{
              display: 'block', padding: '8px 16px',
              fontFamily: 'Inter', fontSize: 13,
              color: active ? '#E6EDF3' : '#6E7681',
              background: active ? 'rgba(45,212,191,0.08)' : 'transparent',
              borderLeft: active ? '2px solid #2DD4BF' : '2px solid transparent',
              textDecoration: 'none', transition: 'all 0.15s',
            }}>
              {label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
