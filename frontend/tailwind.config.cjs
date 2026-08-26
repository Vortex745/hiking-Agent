/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#1a3c2a',
          light: '#2d6a4f',
          hover: '#23543c',
        },
        bg: {
          body: '#f5f5f0',
          card: '#ffffff',
          sidebar: '#1a3c2a',
        },
        text: {
          primary: '#333333',
          secondary: '#666666',
          muted: '#999999',
        },
        border: {
          DEFAULT: '#e8e8e3',
        },
      },
      spacing: {
        sidebar: '220px',
      },
      height: {
        header: '56px',
      },
      borderRadius: {
        sm: '8px',
        md: '12px',
        lg: '16px',
      },
      boxShadow: {
        sm: '0 1px 3px rgba(0,0,0,0.08)',
        md: '0 4px 12px rgba(0,0,0,0.1)',
      },
      animation: {
        shimmer: 'shimmer 2.5s ease-in-out infinite',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '200% center' },
          '100%': { backgroundPosition: '-200% center' },
        },
      },
    },
  },
  plugins: [],
}
