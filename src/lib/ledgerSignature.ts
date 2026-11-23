import type { ResidentInfo } from '$lib/types';

/**
 * A string that changes whenever free graphics memory could have moved
 * because of the ledger: a resident added or removed, a slot swapped to a
 * different model, or a resident's size settling from estimate to measured.
 * Order-independent, so a re-sorted snapshot doesn't count as a change.
 */
export function residentSignature(residents: readonly ResidentInfo[]): string {
	return residents
		.map((r) => `${r.id}=${r.model_id ?? r.label}@${r.actual_mb ?? r.est_mb}`)
		.sort()
		.join('|');
}
