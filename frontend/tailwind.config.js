/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#2C5F2D',
        secondary: '#97BC62',
        accent: '#F9E795',
        dark: '#1a1a2e',
      },
    },
  },
  plugins: [],
}