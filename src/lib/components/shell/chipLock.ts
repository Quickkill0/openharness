/**
 * Decides whether a loaded model may be unloaded right now. Used by every
 * unload control: the dock chips, the Models table, the inspector and
 * Settings > Hardware.
 *
 * Two cases block it:
 * - The slot is being refilled by another model. The orchestrator would run
 *   the unload after that load finishes and unload the newcomer, so the
 *   control reads "Being replaced" instead.
 * - The model is doing work: a chat reply streaming in any conversation, or
 *   a generation job (a Studio page or chat voice) running. Unloading would
 *   kill the sidecar mid-request, so the user is told what to stop first.
 *
 * Activity comes from backend events (see `$lib/activity`), not from what the
 * UI is showing, so switching conversations doesn't release the lock.
 *
 * The backend also refuses an unload whose expected model no longer holds
 * the slot (`orchestrator_unload_if`), which covers anything this misses.
 */
import { slotsForTask, type ChatStreams } from '$lib/activity';
import type { ResidentInfo } from '$lib/types';

export type ChipLock = { kind: 'replacing' | 'in_use'; reason: string } | null;

export interface ChipLockInputs {
	/** Model id the text engine is currently loading, if any. */
	textLoadingModelId: string | null;
	/** Slot id → catalog id for loads started from this window. */
	pendingSlots: ReadonlyMap<string, string>;
	/** Conversation id → text model its reply is streaming from (null = unknown). */
	chatStreams: ChatStreams;
	/** Generation tasks with a job in flight, from Studio pages or chat voice. */
	busyTasks: ReadonlySet<string>;
}

export function chipLock(r: ResidentInfo, s: ChipLockInputs): ChipLock {
	const incoming = s.pendingSlots.get(r.id) ?? (r.id === 'text' ? s.textLoadingModelId : null);
	if (incoming && incoming !== r.model_id) {
		return {
			kind: 'replacing',
			reason: `Another model is loading into this slot and will replace ${r.label}.`,
		};
	}
	if (r.id === 'text') {
		// An unknown model on either side can't be ruled out, so it locks.
		for (const model of s.chatStreams.values()) {
			if (model === null || r.model_id == null || model === r.model_id) {
				return {
					kind: 'in_use',
					reason: 'A chat reply is streaming from this model. Stop it in Chat before unloading.',
				};
			}
		}
	}
	for (const task of s.busyTasks) {
		if (slotsForTask(task).includes(r.id)) {
			return {
				kind: 'in_use',
				reason: 'A generation job is using this model. Let it finish or cancel it before unloading.',
			};
		}
	}
	return null;
}
