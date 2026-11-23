/**
 * Theme store — persisted via tauri-plugin-store to
 * %LOCALAPPDATA%\OpenHarness\settings.json.
 *
 * Uses Svelte 5 runes ($state, $effect) so this is a rune-based reactive
 * module rather than a classic Svelte store.
 */
import { LazyStore } from '@tauri-apps/plugin-store';
import { IS_TAURI } from '$lib/ipc/backend';
import type { Theme } from '$lib/types';

const STORE_FILE = 'settings.json';
const THEME_KEY = 'theme';

// Lazily-initialised plugin-store instance (created once).
// defaults required by StoreOptions type; empty object is valid.
const store = new LazyStore(STORE_FILE, { defaults: {}, autoSave: true });

/** Read the persisted theme, defaulting to 'system'. */
export async function loadTheme(): Promise<Theme> {
	// Outside Tauri (browser dev with the mock backend) plugin-store has no
	// bridge — fall back to localStorage so theming stays testable.
	const raw = IS_TAURI
		? await store.get<string>(THEME_KEY)
		: localStorage.getItem(THEME_KEY);
	if (raw === 'light' || raw === 'dark' || raw === 'system') return raw;
	return 'system';
}

/** Persist the theme and apply it to <html>. */
export async function persistTheme(theme: Theme): Promise<void> {
	if (IS_TAURI) await store.set(THEME_KEY, theme);
	else localStorage.setItem(THEME_KEY, theme);
	applyTheme(theme);
}

/** Apply a theme to the document root immediately (no async). */
export function applyTheme(theme: Theme): void {
	const root = document.documentElement;
	if (theme === 'dark') {
		root.classList.add('dark');
		root.classList.remove('light');
	} else if (theme === 'light') {
		root.classList.remove('dark');
		root.classList.add('light');
	} else {
		// system
		const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
		if (prefersDark) {
			root.classList.add('dark');
			root.classList.remove('light');
		} else {
			root.classList.remove('dark');
			root.classList.add('light');
		}
	}
}
