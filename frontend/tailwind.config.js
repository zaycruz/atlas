/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}'
  ],
  theme: {
    extend: {
      colors: {
        'atlas-black': '#000000',
        'atlas-green-950': '#052E16',
        'atlas-green-900': '#14532D',
        'atlas-green-800': '#166534',
        'atlas-green-700': '#15803D',
        'atlas-green-600': '#16A34A',
        'atlas-green-500': '#22C55E',
        'atlas-green-400': '#4ADE80',
        'atlas-yellow-400': '#FACC15',
        'atlas-cyan-400': '#22D3EE',
        'atlas-cyan-500': '#06B6D4',
        'atlas-red-400': '#F87171'
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'SF Mono', 'Consolas', 'monospace']
      }
    }
  },
  plugins: []
};
