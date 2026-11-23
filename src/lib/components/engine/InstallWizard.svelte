<script lang="ts">
	/**
	 * InstallWizard — modal that walks the user through engine installation.
	 *
	 * Three states driven by engineStore:
	 *  1. Detect    — pre-install: shows GPU summary + variant decision.
	 *  2. Installing — shows pct + stage/bytes from engineStore.status.
	 *  3. Done/Error — auto-closes on success, shows retry on failure.
	 */
	import { onMount, onDestroy } from 'svelte';
	import Button from '$lib/components/ui/Button.svelte';
	import Spinner from '$lib/components/ui/Spinner.svelte';
	import { engineStore, installEngine } from '$lib/stores/engine.svelte';
	import { gpuStore } from '$lib/stores/gpu.svelte';
	import { addToast } from '$lib/stores/toasts.svelte';

	interface Props {
		open: boolean;
		onClose?: () => void;
	}

	let { open = $bindable(false), onClose }: Props = $props();

	/** Track whether the user actually kicked off an install during THIS open. */
	let installStarted = $state(false);
	// Non-reactive: this timer handle is only used internally for clearing. If it
	// were $state, the auto-close $effect below would read AND write it, forming a
	// self-triggering reactive loop that freezes the UI.
	let autoCloseTimer: ReturnType<typeof setTimeout> | null = null;
	let dialogEl = $state<HTMLDivElement | null>(null);

	const status = $derived(engineStore.status);
	const phase = $derived(engineStore.phase);
	const isInstalling = $derived(phase === 'installing');
	// A failure with a launcher is a model load's, not this installer's.
	const isFailed = $derived(status.phase === 'failed' && status.launcher == null);

	const gpuLine = $derived.by(() => {
		const s = gpuStore.status;
		if (s.kind !== 'available') return null;
		return `${s.name} (driver ${s.driver_version}, CUDA ${s.cuda_major}.${s.cuda_minor})`;
	});
	const variantLabel = $derived(gpuStore.hasGpu ? 'CUDA 12.4 build' : 'CPU build');

	function bytesToMB(b: number): string {
		return `${(b / (1024 * 1024)).toFixed(0)} MB`;
	}

	const installingDetails = $derived.by(() => {
		if (status.phase !== 'installing') return null;
		return {
			pct: Math.round(status.pct),
			stage: status.stage,
			bytes: status.bytes,
			total: status.total,
		};
	});

	const failedError = $derived(status.phase === 'failed' && status.launcher == null ? status.error : null);

	function close() {
		if (isInstalling) return; // can't close mid-install
		if (autoCloseTimer) {
			clearTimeout(autoCloseTimer);
			autoCloseTimer = null;
		}
		open = false;
		onClose?.();
	}

	async function startInstall() {
		installStarted = true;
		try {
			await installEngine();
		} catch (e) {
			addToast(`Engine install failed: ${e}`, 'error');
		}
	}

	function retry() {
		installStarted = false;
		startInstall();
	}

	function handleKey(e: KeyboardEvent) {
		if (e.key === 'Escape') close();
	}

	function handleBackdrop() {
		close();
	}

	// Auto-close on successful install.
	$effect(() => {
		// Only auto-close if WE started the install and it has now landed in idle/ready.
		if (!open) return;
		if (!installStarted) return;
		if (isInstalling) return;
		if (isFailed) return;
		// success transition (installing → idle/ready)
		if (phase === 'idle' || phase === 'ready' || phase === 'loading') {
			if (autoCloseTimer) clearTimeout(autoCloseTimer);
			autoCloseTimer = setTimeout(() => {
				open = false;
				onClose?.();
				installStarted = false;
			}, 1500);
		}
	});

	$effect(() => {
		if (open && dialogEl) {
			dialogEl.focus();
		}
		if (!open) {
			installStarted = false;
		}
	});

	onMount(() => {
		document.addEventListener('keydown', handleKey);
	});

	onDestroy(() => {
		document.removeEventListener('keydown', handleKey);
		if (autoCloseTimer) clearTimeout(autoCloseTimer);
	});

	// Showing "Engine ready" success card when we just finished an install we started.
	const showSuccess = $derived(
		installStarted && !isInstalling && !isFailed && (phase === 'idle' || phase === 'ready' || phase === 'loading'),
	);
</script>

{#if open}
	<!-- Backdrop -->
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
		onclick={handleBackdrop}
	>
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			bind:this={dialogEl}
			role="dialog"
			aria-modal="true"
			aria-labelledby="engine-install-title"
			tabindex="-1"
			class="relative w-[min(520px,90vw)] overflow-hidden rounded-xl border border-border bg-surface shadow-2xl focus:outline-none"
			onclick={(e) => e.stopPropagation()}
		>
			<!-- Header -->
			<header class="flex items-center justify-between border-b border-border px-5 py-3">
				<h2 id="engine-install-title" class="text-sm font-semibold text-foreground">
					{#if showSuccess}
						Engine installed
					{:else if isFailed}
						Engine install failed
					{:else if isInstalling}
						Installing engine
					{:else}
						Install OpenHarness engine
					{/if}
				</h2>
				<button
					type="button"
					onclick={close}
					disabled={isInstalling}
					aria-label="Close"
					class="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-elevated hover:text-foreground disabled:opacity-30 disabled:hover:bg-transparent"
				>
					<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="h-4 w-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
					</svg>
				</button>
			</header>

			<!-- Body -->
			<div class="px-5 py-5">
				{#if showSuccess}
					<!-- Success state -->
					<div class="flex flex-col items-center gap-3 py-4 text-center">
						<div class="flex h-12 w-12 items-center justify-center rounded-full bg-success/15 text-success ring-1 ring-success/30">
							<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="h-6 w-6">
								<path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
							</svg>
						</div>
						<p class="text-sm font-medium text-foreground">Engine ready</p>
						<p class="text-xs text-muted-foreground">Closing automatically…</p>
					</div>
				{:else if isFailed}
					<!-- Error state -->
					<div class="space-y-4">
						<div class="rounded-lg border border-error/30 bg-error/10 p-3">
							<p class="text-xs font-medium text-error">Install error</p>
							<p class="mt-1 break-words text-xs text-error/80">
								{failedError ?? engineStore.error ?? 'Unknown error'}
							</p>
						</div>
						<div class="flex justify-end gap-2">
							<Button variant="ghost" size="sm" onclick={close}>Close</Button>
							<Button variant="primary" size="sm" onclick={retry}>Retry</Button>
						</div>
					</div>
				{:else if isInstalling}
					<!-- Installing state -->
					<div class="space-y-4">
						<div class="space-y-2">
							<div class="flex items-center justify-between text-xs">
								<span class="font-medium text-foreground">
									{installingDetails?.stage ?? 'Starting…'}
								</span>
								<span class="tabular-nums text-muted-foreground">
									{installingDetails?.pct ?? 0}%
								</span>
							</div>
							<div class="h-2 w-full overflow-hidden rounded-full bg-elevated ring-1 ring-border/60">
								<div
									class="h-full rounded-full bg-primary transition-all duration-300"
									style="width: {installingDetails?.pct ?? 0}%"
								></div>
							</div>
							{#if installingDetails && installingDetails.total > 0}
								<p class="text-[11px] tabular-nums text-muted-foreground">
									{bytesToMB(installingDetails.bytes)} / {bytesToMB(installingDetails.total)}
								</p>
							{/if}
						</div>
						<div class="flex justify-end gap-2">
							<Button variant="ghost" size="sm" disabled>Cancel</Button>
						</div>
					</div>
				{:else}
					<!-- Detect state -->
					<div class="space-y-4">
						<p class="text-sm text-muted-foreground">
							OpenHarness bundles llama.cpp on first run. This downloads about 700 MB to
							<code class="rounded bg-elevated px-1 py-0.5 text-[11px] text-foreground">%LOCALAPPDATA%\OpenHarness\engine</code>.
						</p>

						<div class="rounded-lg border border-border bg-elevated/40 p-3">
							{#if gpuStore.hasGpu && gpuLine}
								<div class="flex items-start gap-2.5">
									<span class="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-success/15 text-success ring-1 ring-success/30">
										<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="h-3 w-3">
											<path stroke-linecap="round" stroke-linejoin="round" d="M4.5 12.75l6 6 9-13.5" />
										</svg>
									</span>
									<div class="min-w-0">
										<p class="text-xs font-medium text-foreground">Detected NVIDIA GPU</p>
										<p class="mt-0.5 break-words text-xs text-muted-foreground">{gpuLine}</p>
										<p class="mt-1 text-[11px] text-muted-foreground">
											→ <span class="text-primary">CUDA 12.4 build</span> will be installed
										</p>
									</div>
								</div>
							{:else}
								<div class="flex items-start gap-2.5">
									<span class="mt-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-muted text-muted-foreground">
										<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="h-3 w-3">
											<path fill-rule="evenodd" d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12zm8.706-1.442c1.146-.573 2.437.463 2.126 1.706l-.709 2.836.042-.02a.75.75 0 01.67 1.34l-.04.022c-1.147.573-2.438-.463-2.127-1.706l.71-2.836-.042.02a.75.75 0 11-.671-1.34l.041-.022zM12 9a.75.75 0 100-1.5.75.75 0 000 1.5z" clip-rule="evenodd" />
										</svg>
									</span>
									<div class="min-w-0">
										<p class="text-xs font-medium text-foreground">No NVIDIA GPU detected</p>
										<p class="mt-0.5 text-xs text-muted-foreground">
											Inference will run on CPU — works fine, just slower.
										</p>
										<p class="mt-1 text-[11px] text-muted-foreground">
											→ <span class="text-foreground">CPU build</span> will be installed
										</p>
									</div>
								</div>
							{/if}
						</div>

						<div class="flex justify-end gap-2 pt-1">
							<Button variant="ghost" size="sm" onclick={close}>Cancel</Button>
							<Button
								variant="primary"
								size="md"
								onclick={startInstall}
							>
								<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="h-3.5 w-3.5">
									<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
								</svg>
								Install engine (~700 MB)
							</Button>
						</div>
					</div>
				{/if}
			</div>

			<!-- Live status footer (only when relevant) -->
			{#if isInstalling}
				<footer class="flex items-center gap-2 border-t border-border bg-surface/60 px-5 py-2">
					<Spinner size="sm" />
					<span class="text-[11px] text-muted-foreground">Don't close the app — install can take a few minutes.</span>
				</footer>
			{/if}
		</div>
	</div>
{/if}
