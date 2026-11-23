/**
 * Shared UI state — used to communicate between child routes and the layout.
 * Follows the single-writer pattern: each piece of state has exactly one setter.
 */

let _downloadsOpen = $state(false);

export const uiStore = {
	get downloadsOpen() {
		return _downloadsOpen;
	},
	set downloadsOpen(v: boolean) {
		_downloadsOpen = v;
	},
};

export function openDownloads(): void {
	_downloadsOpen = true;
}

export function closeDownloads(): void {
	_downloadsOpen = false;
}

// --- Dock "will load" preview --------------------------------------------

/**
 * A ghost segment the bottom dock draws on the graphics-memory meter: what
 * loading the model selected on the Models page would add, from today's
 * `estimate_model_fit`. The Models page is the only writer; it clears the
 * preview when the selection changes, the model loads, or the page unmounts.
 */
export interface DockPreview {
	/** Catalog id the estimate belongs to (guards against stale writes). */
	modelId: string;
	/** Short display name for the dock caption. */
	label: string;
	/** Estimated graphics-memory cost of loading it, in MiB. */
	vramMb: number;
	/**
	 * Resident the load would replace (a chat load swaps the llama.cpp slot),
	 * and the MiB that frees first. Null/0 when nothing is replaced.
	 */
	replacesId: string | null;
	freesMb: number;
}

let _dockPreview = $state<DockPreview | null>(null);

export const dock = {
	get preview() {
		return _dockPreview;
	},
};

export function setDockPreview(p: DockPreview | null): void {
	_dockPreview = p;
}
