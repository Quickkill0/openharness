/**
 * Minimal toast notification store.
 * Components subscribe to `toasts` to render active notifications.
 * Use `addToast()` to push new messages; they auto-dismiss after `duration` ms.
 */

export type ToastVariant = 'info' | 'success' | 'warning' | 'error';

export interface Toast {
	id: string;
	message: string;
	variant: ToastVariant;
	/** Auto-dismiss duration in milliseconds (0 = persistent until dismissed). */
	duration: number;
}

let _toasts = $state<Toast[]>([]);

export const toasts = {
	get value() {
		return _toasts;
	},
};

let _counter = 0;

export function addToast(
	message: string,
	variant: ToastVariant = 'info',
	duration = 4000,
): string {
	const id = `toast-${++_counter}`;
	_toasts = [..._toasts, { id, message, variant, duration }];

	if (duration > 0) {
		setTimeout(() => removeToast(id), duration);
	}

	return id;
}

export function removeToast(id: string): void {
	_toasts = _toasts.filter((t) => t.id !== id);
}
