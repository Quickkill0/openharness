/**
 * Shared Python-worker status store.
 *
 * The Rust side keeps a SINGLE worker-status subscriber slot (a new
 * `subscribe_worker_status` call aborts the previous forwarder), so every
 * page subscribing on its own clobbers the others — Studio's banner froze
 * whenever Settings → Runtime subscribed after it. All consumers read this
 * one store instead; exactly one channel subscription is ever made.
 */
import { subscribeWorkerStatus } from '$lib/ipc/channels';
import type { WorkerStatus } from '$lib/types';

let _status = $state<WorkerStatus>({ phase: 'not_installed' });
let wired = false;

export const workerStore = {
	get status(): WorkerStatus {
		return _status;
	},
};

/** Idempotent: subscribe the single worker-status channel. */
export async function initWorkerStore(): Promise<void> {
	if (wired) return;
	wired = true;
	await subscribeWorkerStatus((s) => {
		_status = s;
	});
}
