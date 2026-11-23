import type { Config } from 'tailwindcss';

// Tailwind 4 uses CSS-first config via @theme in app.css.
// This file is kept for tooling compatibility (e.g. IDE plugins).
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	darkMode: 'class',
} satisfies Config;
