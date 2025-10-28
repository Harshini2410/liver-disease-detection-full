/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0faf9',
          100: '#dcf3f0',
          200: '#bae7e1',
          300: '#8ed3cb',
          400: '#5eb9af',
          500: '#3a9d91', // Modern teal
          600: '#2c7c73',
          700: '#236159',
          800: '#1a4740',
          900: '#0e2623',
        },
        secondary: {
          50: '#fdf2f4',
          100: '#fbe7eb',
          200: '#f7cfd8',
          300: '#f2a7b8',
          400: '#eb7d95',
          500: '#e54d6e', // Fresh rose
          600: '#d1325c',
          700: '#ab2047',
          800: '#8c1937',
          900: '#591021',
        },
        medical: {
          fibrosis: '#4b96a3', // Ocean blue
          steatosis: '#50a186', // Mint
          inflammation: '#e57f98', // Soft rose
          ballooning: '#7c98b3', // Steel blue
          success: '#50a186', // Mint
          warning: '#e3aa4f', // Amber
          error: '#e57373', // Soft red
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
