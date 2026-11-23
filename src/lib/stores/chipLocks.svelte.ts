/**
 * Live unload locks for ledger residents, shared by every unload control
 * (dock chips, Models table, inspector, Settings > Hardware). The rules live
 * in the pure `chipLock`; this feeds it from the engine and ledger stores and
 * from the event-driven activity the chat and Studio stores track.
 */
import { chipLock, type ChipLock } from '$lib/components/shell/chipLock';
import { chatStore } from '$lib/stores/chat.svelte';
import { engineStore } from '$lib/stores/engine.svelte';
import { ledgerStore } from '$lib/stores/ledger.svelte';
import { studio } from '$lib/stores/studio.svelte';
import type { ResidentInfo } from '$lib/types';

/** Why this resident can't be unloaded right now, or null. Reactive. */
export function unloadLock(r: ResidentInfo): ChipLock {
	const status = engineStore.status;
	return chipLock(r, {
		textLoadingModelId: status.phase === 'loading' ? status.model_id : null,
		pendingSlots: ledgerStore.pendingSlots,
		chatStreams: chatStore.inflight,
		busyTasks: studio.busyTasks,
	});
}
