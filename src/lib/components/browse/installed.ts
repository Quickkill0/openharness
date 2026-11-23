/**
 * "Is this already installed?" checks for the Store inspector. Pure so the
 * identity rules are unit-tested; the callers pass catalog rows already
 * filtered to the repo on screen.
 */
import type { InstalledModel } from '$lib/types';

type Row = Pick<InstalledModel, 'id' | 'primary_rel_path' | 'primary_sha256' | 'local_path' | 'file_count'>;

function basename(path: string): string {
	return (path.split(/[\\/]/).pop() ?? path).toLowerCase();
}

function normSha(sha: string | null | undefined): string | null {
	return sha?.trim().toLowerCase() || null;
}

/**
 * A repo file counts as installed when a catalog row is that exact file.
 *
 * When Hugging Face gives the file's SHA-256, the hash is the only test: it
 * must equal the row's primary-file hash or its id (single GGUFs use the hash
 * as id). Otherwise the file name must equal the row's original repo
 * filename (`primary_rel_path`), never the content-addressed name on disk.
 * Quant labels are never identity: two files in one repo can share "Q4_K_M".
 */
export function isFileInstalled(file: { path: string; sha256?: string | null }, installed: Row[]): boolean {
	const sha = normSha(file.sha256);
	if (sha) {
		return installed.some((m) => normSha(m.primary_sha256) === sha || m.id.toLowerCase() === sha);
	}
	const name = basename(file.path);
	return installed.some((m) => m.primary_rel_path != null && basename(m.primary_rel_path) === name);
}

/**
 * True when the repo's runnable folder is installed in full: a dir-layout
 * catalog row (it has a `local_path`) holding at least as many files as
 * "Download repo" would fetch. A lone GGUF or a partial folder from the
 * same repo does not count.
 */
export function isRepoInstallComplete(installed: Row[], subsetCount: number): boolean {
	if (subsetCount <= 0) return false;
	return installed.some((m) => m.local_path !== null && m.file_count >= subsetCount);
}
