import { serverRequestLog, serverStatus, serverSubscribeRequestLog, type ServerRequestLog, type ServerStatus } from '$lib/ipc/server';

export const serverLog = $state<{ entries: ServerRequestLog[] }>({ entries: [] });
let subscription: Promise<void> | undefined;
// One app-lifetime channel, even when Settings is revisited.
export function initServerLog(): Promise<void> {
    if (!subscription) subscription = (async () => {
        const pending: ServerRequestLog[] = [];
        let ready = false;
        await serverSubscribeRequestLog(entry => {
            if (!ready) pending.push(entry);
            else serverLog.entries = [...serverLog.entries, entry].slice(-500);
        });
        const snapshot = await serverRequestLog();
        const merged = [...snapshot, ...pending];
        serverLog.entries = merged.filter((entry, index) => merged.findIndex(other => JSON.stringify(other) === JSON.stringify(entry)) === index).slice(-500);
        ready = true;
    })().catch(error => { subscription = undefined; throw error; });
    return subscription;
}

// Last known server status, shared by the Server page and the dock's server pill.
export const serverState = $state<{ status: ServerStatus | null }>({ status: null });
export function setServerStatus(status: ServerStatus): void {
    serverState.status = status;
}
export async function refreshServerStatus(): Promise<void> {
    try {
        serverState.status = await serverStatus();
    } catch {
        // Keep the last known status; the Server page shows the error.
    }
}
