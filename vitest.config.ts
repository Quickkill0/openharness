import { defineConfig } from 'vitest/config';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { svelteTesting } from '@testing-library/svelte/vite';
import { fileURLToPath } from 'node:url';

// Standalone vitest config (no SvelteKit plugin needed). The Svelte plugin
// compiles `.svelte.ts` rune stores so their state flows can be tested
// against the mock IPC backend; pure modules don't need it. svelteTesting
// resolves Svelte's browser build and cleans up the DOM, so component tests
// (files marked `// @vitest-environment jsdom`) can mount real components.
export default defineConfig({
	plugins: [svelte({ configFile: false }), svelteTesting()],
	resolve: { alias: { $lib: fileURLToPath(new URL('./src/lib', import.meta.url)) } },
	test: {
		include: ['src/**/*.test.ts'],
		environment: 'node',
	},
});
