<script lang="ts">
	/** Settings, Hugging Face token: stored in Windows Credential Manager. */
	import { onMount } from 'svelte';
	import Button from '$lib/components/ui/Button.svelte';
	import Input from '$lib/components/ui/Input.svelte';
	import { getHfToken, setHfToken, clearHfToken, validateHfToken } from '$lib/ipc/commands';

	let input = $state('');
	let saved = $state(false);
	let valid = $state<boolean | null>(null);
	let user = $state<string | null>(null);
	let error = $state<string | null>(null);
	let busy = $state(false);
	/** True while the user is replacing a stored token. */
	let replacing = $state(false);

	onMount(async () => {
		try {
			const stored = await getHfToken();
			if (stored && stored.length > 0) {
				saved = true;
				await validate();
			}
		} catch (e) {
			error = String(e);
		}
	});

	async function validate() {
		busy = true;
		error = null;
		valid = null;
		user = null;
		try {
			const who = await validateHfToken();
			valid = true;
			user = who.name;
		} catch (e) {
			valid = false;
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}

	async function save() {
		if (!input.trim()) return;
		busy = true;
		error = null;
		try {
			await setHfToken(input.trim());
			saved = true;
			replacing = false;
			input = '';
			await validate();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			busy = false;
		}
	}

	async function remove() {
		busy = true;
		try {
			await clearHfToken();
			saved = false;
			valid = null;
			user = null;
			error = null;
			input = '';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			busy = false;
		}
	}
</script>

<div class="wb-set-h">
	<h2>Hugging Face token</h2>
	<p>Needed only for gated models and higher download limits</p>
</div>

<div class="wb-prop">
	<div class="pl">
		Access token
		<small>Stored in Windows Credential Manager</small>
	</div>
	<div class="pv">
		{#if saved && !replacing}
			<div class="flex flex-wrap items-center gap-2">
				<span class="wb-code select-none tracking-widest text-muted-foreground">hf_••••••••••••••••</span>
				<Button variant="secondary" size="sm" onclick={() => (replacing = true)} disabled={busy}>Replace</Button>
				<Button variant="secondary" size="sm" onclick={validate} disabled={busy}>Check again</Button>
				<Button variant="danger" size="sm" onclick={remove} disabled={busy}>Remove</Button>
			</div>
		{:else}
			<form
				class="tok-form flex max-w-md gap-2"
				onsubmit={(e) => {
					e.preventDefault();
					void save();
				}}
			>
				<label for="hf-token" class="sr-only">Hugging Face access token</label>
				<Input id="hf-token" type="password" bind:value={input} placeholder="hf_…" autocomplete="off" disabled={busy} />
				<Button variant="primary" size="md" type="submit" disabled={busy || !input.trim()}>
					{busy ? 'Saving…' : 'Save'}
				</Button>
				{#if replacing}
					<Button variant="ghost" size="md" onclick={() => { replacing = false; input = ''; }}>Cancel</Button>
				{/if}
			</form>
		{/if}
	</div>
</div>

{#if saved}
	<div class="wb-prop">
		<div class="pl">Signed in as</div>
		<div class="pv flex items-center gap-2">
			{#if valid === true}
				<span class="badge ok">Valid</span>
				<span>{user ?? 'Unknown account'}</span>
			{:else if valid === false}
				<span class="badge bad">Not accepted</span>
			{:else}
				<span class="text-muted-foreground">Checking…</span>
			{/if}
		</div>
	</div>
{/if}

{#if error}
	<p class="px-5 pt-3 text-xs text-error" role="alert">{error}</p>
{/if}

<style>
	/* Keep the buttons at their natural width; the input takes the slack. */
	.tok-form :global(button) {
		flex-shrink: 0;
		white-space: nowrap;
	}
	.badge {
		display: inline-flex;
		align-items: center;
		height: 20px;
		padding: 0 8px;
		border-radius: 10px;
		font-size: var(--text-2xs);
		font-weight: 600;
	}
	.badge.ok {
		background: color-mix(in oklch, var(--success) 16%, transparent);
		color: var(--success);
	}
	.badge.bad {
		background: color-mix(in oklch, var(--error) 16%, transparent);
		color: var(--error);
	}
</style>
