/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/web/**/*.html"],
  theme: {
    extend: {
      colors: {
        parchment: { 50: "#faf8f3", 100: "#f5f0e6", 200: "#e8dfc9" },
        ink: { DEFAULT: "#2c2a24", light: "#6b6557", muted: "#8c7d62" },
        accent: { DEFAULT: "#5a4e38", hover: "#3d3424" },
      },
      fontFamily: {
        serif: ["Palatino", '"Book Antiqua"', "serif"],
        mono: ["ui-monospace", "SFMono-Regular", "monospace"],
      },
    },
  },
  plugins: [],
};
