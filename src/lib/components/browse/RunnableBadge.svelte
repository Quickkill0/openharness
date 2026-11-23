<script lang="ts">
	/**
	 * RunnableBadge — honest "can OpenHarness run this?" verdict (M9 §9.5).
	 *
	 * Three states from a Runnability verdict:
	 *  - runnable + available  → green "Runs here" (with runtime detail)
	 *  - runnable + !available → yellow (pipeline exists but deps missing)
	 *  - !runnable             → slate "Not runnable" (download-only)
	 *
	 * `detailed` renders the runtime/pipeline name; compact rows show a dot+word.
	 */
	import type { Runnability } from '$lib/types';

	interface Props {
		verdict?: Runnability | null;
		/** Show runtime name + tooltip copy inline (detail pane). */
		detailed?: boolean;
	}

	let { verdict = null, detailed = false }: Props = $props();

	const tone = $derived(
		!verdict
			? null
			: verdict.runnable && verdict.available
				? 'ok'
				: verdict.runnable
					? 'warn'
					: 'no',
	);

	const label = $derived.by(() => {
		if (!verdict) return '';
		if (!verdict.runnable) return detailed ? "Won't run on this PC" : "Won't run here";
		const rt = verdict.runtime === 'llama-cpp' ? 'llama.cpp' : 'Worker';
		if (!detailed) return verdict.available ? 'Runs on this PC' : 'Needs setup';
		const via = verdict.pipeline ? `${rt} · ${verdict.pipeline}` : rt;
		return verdict.available ? `Runs on this PC — ${via}` : `Needs setup — ${via}`;
	});
</script>

{#if verdict && tone}
	<span class="wb-badge {tone === 'ok' ? 'ok' : tone === 'warn' ? 'warn' : 'err'}" title={verdict.reason}>
		{label}
	</span>
{/if}
