/**
 * Number formatting shared by the Workbench shell (dock meters, Models table,
 * inspector). Memory figures from the backend arrive in MiB; sizes on disk
 * arrive in bytes. Both render as binary gigabytes with one decimal, which is
 * how Windows Task Manager and nvidia-smi present them.
 */

const MIB_PER_GIB = 1024;
const BYTES_PER_GIB = 1024 ** 3;

/** MiB → "12.4" (no unit). Values under 0.1 GB show one more decimal. */
export function gbFromMb(mb: number): string {
	const gb = Math.max(0, mb) / MIB_PER_GIB;
	if (gb > 0 && gb < 0.1) return gb.toFixed(2);
	return gb.toFixed(1);
}

/** Bytes → "4.9 GB" / "820 MB" / "12 KB". */
export function formatBytes(bytes: number | null | undefined): string {
	if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return '—';
	if (bytes >= BYTES_PER_GIB) return `${(bytes / BYTES_PER_GIB).toFixed(1)} GB`;
	if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`;
	return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Fraction of `total` as a CSS percentage clamped to [0, 100]. */
export function pct(part: number, total: number): string {
	if (!(total > 0)) return '0%';
	return `${Math.min(100, Math.max(0, (part / total) * 100)).toFixed(2)}%`;
}

/** Large counts → "412", "1.3K", "2.4M" (Store downloads and likes). */
export function compactCount(n: number | null | undefined): string {
	if (n == null || !Number.isFinite(n) || n < 0) return '—';
	if (n >= 1_000_000) return `${trimZero((n / 1_000_000).toFixed(1))}M`;
	if (n >= 1_000) return `${trimZero((n / 1_000).toFixed(1))}K`;
	return String(Math.round(n));
}

function trimZero(s: string): string {
	return s.endsWith('.0') ? s.slice(0, -2) : s;
}

/**
 * ISO timestamp → plain "3 days ago" / "2 weeks ago" / "today". Returns
 * null for missing or unparseable input so callers can hide the cell.
 */
export function timeAgo(iso: string | null | undefined, now: number = Date.now()): string | null {
	if (!iso) return null;
	const t = Date.parse(iso);
	if (!Number.isFinite(t)) return null;
	const days = Math.floor(Math.max(0, now - t) / 86_400_000);
	if (days < 1) return 'today';
	if (days === 1) return 'yesterday';
	if (days < 7) return `${days} days ago`;
	if (days < 30) {
		const w = Math.floor(days / 7);
		return w === 1 ? '1 week ago' : `${w} weeks ago`;
	}
	if (days < 365) {
		const m = Math.floor(days / 30);
		return m === 1 ? '1 month ago' : `${m} months ago`;
	}
	const y = Math.floor(days / 365);
	return y === 1 ? '1 year ago' : `${y} years ago`;
}
