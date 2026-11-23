<script lang="ts">
    import { onMount } from 'svelte';
    import Card from '$lib/components/ui/Card.svelte';
    import { serverStatus, serverStart, serverStop, serverUpdateSettings, serverGetKey, serverRegenerateKey, serverConnectionInfo, type ServerStatus, type ServerConnectionInfo } from '$lib/ipc/server';
    import { initServerLog, serverLog, setServerStatus } from '$lib/stores/server.svelte';

    let status = $state<ServerStatus | null>(null);
    let connections = $state<ServerConnectionInfo | null>(null);
    let key = $state<string | null>(null);
    let busy = $state(false);
    let error = $state('');
    let notice = $state('');
    const inputClass = 'w-full rounded-md border border-border bg-surface px-3 py-2 text-sm';
    const buttonClass = 'rounded-md border border-border px-3 py-2 text-sm disabled:opacity-50';

    async function action(fn: () => Promise<void>) {
        busy = true; error = ''; notice = '';
        try { await fn(); } catch (e) { error = String(e); } finally { busy = false; }
    }
    async function refresh(next?: ServerStatus) {
        status = next ?? await serverStatus();
        setServerStatus(status);
        connections = await serverConnectionInfo();
    }
    onMount(() => { void action(async () => { await refresh(); await initServerLog(); }); });
    async function save() {
        if (!status) return;
        const settings = $state.snapshot(status.settings);
        if (!Number.isInteger(settings.port) || settings.port < 1 || settings.port > 65535 || !Number.isInteger(settings.idle_minutes) || settings.idle_minutes < 0) throw new Error('Enter a valid port and idle timeout.');
        await refresh(await serverUpdateSettings(settings));
        notice = 'Server settings saved.';
    }
</script>

<section aria-labelledby="server-title">
    <Card>
        <h2 id="server-title" class="mb-2 text-base font-semibold">OpenAI-compatible server</h2>
        <p class="mb-4 text-sm text-muted-foreground">Connect AI Shuffle or another OpenAI-compatible client to your local text models.</p>
        {#if error}<p role="alert" class="mb-3 text-sm text-error">{error}</p>{/if}
        {#if notice}<p role="status" class="mb-3 text-sm text-muted-foreground">{notice}</p>{/if}
        {#if status}
            <div class="mb-4 flex items-center justify-between gap-3">
                <span class="text-sm">Server: <strong>{status.running ? 'Running' : 'Stopped'}</strong></span>
                <button class={buttonClass} disabled={busy} onclick={() => action(async () => { await refresh(await (status?.running ? serverStop() : serverStart())); })}>{status.running ? 'Stop server' : 'Start server'}</button>
            </div>
            {#if status.error}<p role="alert" class="mb-3 text-sm text-error">{status.error}</p>{/if}
            <form class="space-y-4" onsubmit={(event) => { event.preventDefault(); void action(save); }}>
                <fieldset disabled={busy} class="space-y-4">
                    <div class="grid grid-cols-2 gap-4">
                        <label class="space-y-1 text-sm">Port<input class={inputClass} type="number" min="1" max="65535" required bind:value={status.settings.port} /></label>
                        <label class="space-y-1 text-sm">Listen on<select class={inputClass} bind:value={status.settings.bind}><option value="this_pc">This PC only</option><option value="home_network">Home network</option></select></label>
                    </div>
                    {#if status.settings.bind === 'home_network'}
                        <p class="text-sm text-muted-foreground">Windows may ask to allow OpenHarness through the firewall. Allow private networks only. Home network always requires the API key.</p>
                        <p class="text-xs text-muted-foreground">This API uses HTTP. Use it only on a trusted home network, not a public network or a port-forwarded internet connection.</p>
                    {/if}
                    <label class="flex items-center gap-2 text-sm"><input type="checkbox" bind:checked={status.settings.require_key} disabled={status.settings.bind === 'home_network'} />Require API key on this PC too</label>
                    <label class="flex items-center gap-2 text-sm"><input type="checkbox" bind:checked={status.settings.auto_load} />Auto-load installed text models on request</label>
                    <label class="block space-y-1 text-sm">Unload API-used models after idle minutes (0 = never)<input class={inputClass} type="number" min="0" max="525600" required bind:value={status.settings.idle_minutes} /></label>
                    <details>
                        <summary class="cursor-pointer text-sm">Model API aliases</summary>
                        <div class="mt-3 space-y-2">
                            {#each status.models as model (model.id)}
                                <label class="block text-sm">{model.name}<input class={inputClass} aria-label={'API alias for ' + model.name} placeholder={model.alias} value={status.settings.aliases[model.id] ?? model.alias} onchange={(event) => { if (status) status.settings.aliases[model.id] = event.currentTarget.value; }} /></label>
                            {/each}
                            <p class="text-xs text-muted-foreground">Use a unique lowercase slug, such as qwen-coder. Model IDs also work.</p>
                        </div>
                    </details>
                    <button type="submit" class={buttonClass}>Save server settings</button>
                </fieldset>
            </form>
            <div class="mt-5 space-y-3 border-t border-border pt-4">
                <p class="text-sm font-medium">Access key</p>
                <p class="break-all font-mono text-xs" aria-label="API access key">{key ?? 'Hidden — stored in Windows Credential Manager'}</p>
                <div class="flex flex-wrap gap-2">
                    <button class={buttonClass} disabled={busy} onclick={() => action(async () => { key = key ? null : await serverGetKey(); })}>{key ? 'Hide key' : 'Show key'}</button>
                    <button class={buttonClass} disabled={busy} onclick={() => action(async () => { const value = await serverGetKey(); if (value) { await navigator.clipboard.writeText(value); notice = 'API key copied.'; } })}>Copy key</button>
                    <button class={buttonClass} disabled={busy} onclick={() => action(async () => { await serverRegenerateKey(); key = null; notice = 'API key regenerated. Update your clients; the old key no longer works.'; })}>Regenerate key</button>
                </div>
            </div>
            {#if connections}
                <div class="mt-5 space-y-2 border-t border-border pt-4 text-sm">
                    <h3 class="font-medium">Client connection URLs (saved settings)</h3>
                    <p class="break-all font-mono text-xs">This PC: {connections.this_pc}</p>
                    {#each connections.lan as url}<p class="break-all font-mono text-xs">Home network: {url}</p>{/each}
                    <p class="text-muted-foreground">{connections.auth_required ? 'Authorization: Bearer <your API key>' : 'No key required. Clients that insist on a key can use the generated API key.'}</p>
                    {#if !status.running}<p class="text-muted-foreground">Start the server before connecting.</p>{/if}
                </div>
            {/if}
            <div class="mt-5 border-t border-border pt-4">
                <h3 class="mb-2 text-sm font-medium">Recent requests</h3>
                <p class="mb-2 text-xs text-muted-foreground">Last 500 requests, kept in memory only. No prompts or responses are logged. Tokens appear when the model server reports usage.</p>
                <div class="max-h-64 overflow-auto">
                    <table class="w-full whitespace-nowrap text-left text-xs">
                        <thead><tr><th class="p-2">Time</th><th class="p-2">Client IP</th><th class="p-2">Model</th><th class="p-2">In / out</th><th class="p-2">tok/s</th><th class="p-2">Status</th><th class="p-2">Duration</th></tr></thead>
                        <tbody>{#each [...serverLog.entries].reverse() as entry}<tr class="border-t border-border"><td class="p-2">{new Date(entry.time).toLocaleTimeString()}</td><td class="p-2">{entry.client_ip}</td><td class="max-w-36 truncate p-2" title={entry.model ?? ''}>{entry.model ?? '—'}</td><td class="p-2">{entry.prompt_tokens ?? '—'} / {entry.completion_tokens ?? '—'}</td><td class="p-2">{entry.tokens_per_second?.toFixed(1) ?? '—'}</td><td class="p-2">{entry.status}</td><td class="p-2">{entry.duration_ms} ms</td></tr>{/each}</tbody>
                    </table>
                    {#if serverLog.entries.length === 0}<p class="p-2 text-xs text-muted-foreground">No requests yet.</p>{/if}
                </div>
            </div>
        {/if}
    </Card>
</section>
