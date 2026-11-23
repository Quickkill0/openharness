import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],

	// Tauri expects a fixed port and does not tolerate Vite opening a browser
	server: {
		port: 1420,
		strictPort: true,
		watch: {
			// Background-agent git worktrees live under .claude/worktrees — their
			// file churn must not reload the dev app or its tsconfig cache.
			ignored: ['**/.claude/**'],
		},
	},

	// Prevent Vite from obscuring Rust errors
	clearScreen: false,

	// Tauri on Windows does not support the Vite HMR overlay
	build: {
		// Tauri uses Chromium on Windows so ES2021 is safe
		target: ['es2021', 'chrome100'],
		minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
		sourcemap: !!process.env.TAURI_DEBUG,
	},
});
