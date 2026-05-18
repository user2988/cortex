import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#0D1117',
        card:    '#161B22',
        border:  '#21262D',
        'text-primary':   '#E6EDF3',
        'text-secondary': '#8B949E',
        'text-muted':     '#6E7681',
        'text-dim':       '#484F58',
        teal:    '#2DD4BF',
        blue:    '#4A90D9',
        purple:  '#8B5CF6',
        green:   '#10B981',
        amber:   '#F59E0B',
        red:     '#EF4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
