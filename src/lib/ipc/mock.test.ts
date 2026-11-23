import { describe, expect, it, vi } from 'vitest';
import { mockInvoke, mockSubscribe } from './mock';
import type {
	ChatEvent,
	ContextMemory,
	ConversationSummaryDto,
	FitEstimate,
	InstalledModel,
	LedgerSnapshot,
	LoadPlan,
	MessageDto,
	ModelLoadOutcome,
	PlanOverrides,
} from '$lib/types';

// A fresh mock with nothing loaded, so the card has room for the dense layers
// and an expert tier that seeds at least as fast as the CPU layout's.
async function emptyCard() {
	vi.resetModules();
	const fresh = await import('./mock');
	for (const residentId of ['text', 'pipeline:text_to_speech'])
		await fresh.mockInvoke('orchestrator_unload', { residentId });
	return fresh.mockInvoke;
}

describe('model-runner mock backend', () => {
	it('streams a plain chat reply and persists both messages', async () => {
		vi.useFakeTimers();
		try {
			const conversation = await mockInvoke<ConversationSummaryDto>('create_conversation');
			const events: ChatEvent[] = [];
			mockSubscribe<ChatEvent>('subscribe_chat_events', (event) => {
				if (event.conv_id === conversation.id) events.push(event);
			});
			await mockInvoke('send_message', {
				convId: conversation.id,
				content: 'Hello local model',
			});
			await vi.runAllTimersAsync();
			expect(events[0].type).toBe('user_message_stored');
			expect(events.some((event) => event.type === 'token')).toBe(true);
			expect(events.at(-1)?.type).toBe('assistant_done');
			const messages = await mockInvoke<MessageDto[]>('get_conversation', {
				id: conversation.id,
			});
			expect(messages.map((message) => message.role)).toEqual(['user', 'assistant']);
			expect(messages[0].content).toBe('Hello local model');
			expect(messages[1].content).toContain('mock streamed reply');
			await mockInvoke('delete_conversation', { id: conversation.id });
		} finally {
			vi.useRealTimers();
		}
	});

	it('installed GLM fit uses native tiers and supplied previews are re-planned', async () => {
		const modelId = 'glm-5.3-flash-ud-q2_k_xl';
		const fit = await mockInvoke<FitEstimate>('estimate_model_fit', { modelId, ctx: 8192 });
		expect(fit.plan?.engine_kind).toBe('ssd_stream');
		expect(fit.verdict).toBe('fits');
		// The loaded Qwen2.5 model is replaced, so its graphics memory counts as
		// free and the context cache sits on the card.
		expect(fit.kv_at_ctx_mb).toBeGreaterThan(0);
		expect(fit.estimated_mb).toBeLessThan(24 * 1024);
		const forged = { ...fit.plan!, feasible: true, ctx: 1, overrides: { engine: 'llama_server' as const } };
		await expect(mockInvoke('load_model', { modelId, plan: forged, force: true })).rejects.toThrow();
	});

	it('plans GLM-5.3-Flash across all three tiers on the 4090 profile', async () => {
		const plan = await (await emptyCard())<LoadPlan>('plan_load', {
			modelId: 'glm-5.3-flash-ud-q2_k_xl',
		});
		expect(plan.engine_kind).toBe('ssd_stream');
		expect(plan.tier_mix).toBe('ssd');
		expect(plan.feasible).toBe(true);
		// With the card empty the dense layers and the context cache run on it;
		// experts stay on the host.
		expect(plan.gpu_layers).toBe(-1);
		expect(plan.tiers.vram_weights).toBeGreaterThan(0);
		expect(plan.tiers.vram_kv).toBeGreaterThan(0);
		expect(plan.args[plan.args.indexOf('-ngl') + 1]).toBe('all');
		expect(plan.args).toContain('--cpu-moe');
		expect(plan.tiers.ram + 12 * 1024 ** 3).toBeLessThanOrEqual(48_000_000_000);
		expect(plan.native!.vram_reserve_mib).toBe(5120);
		expect(plan.native!.prefetch_depth).toBe(4);
		expect(plan.args).toContain(`host.budget_mib=${plan.native!.host_budget_mib}`);
		expect(plan.native!.ram_budget_mib).toBeGreaterThan(0);
		expect(plan.speed.decode_tps).toBeGreaterThan(0);
		expect(plan.speed.decode_tps).toBeLessThanOrEqual(3.5);
		expect(plan.speed.prompt_tps).toBe(0); // Unknown, not a GPU-prefill claim.
		expect(plan.speed.label).toBe('slow');
		expect(plan.args).toContain('--ssd');
		for (const flag of ['--no-op-offload', '--no-kv-offload', '--n-cpu-moe', '--moe-cache-mib', '--fit', '--nextn'])
			expect(plan.args).not.toContain(flag);
		expect(plan.native!.estimate_basis).toContain('3.5 tok/s');
		// With Qwen2.5 and Kokoro loaded: the load replaces Qwen2.5, so its
		// graphics memory counts as free, but Kokoro keeps its charge.
		const loaded = await mockInvoke<LoadPlan>('plan_load', {
			modelId: 'glm-5.3-flash-ud-q2_k_xl',
		});
		expect(loaded.gpu_layers).toBe(-1);
		expect(plan.tiers.vram_expert_cache - loaded.tiers.vram_expert_cache).toBe(1180 * 1024 * 1024);
		// A limit whose tier beside the dense layers would seed slower than the
		// larger tier alone keeps them on the CPU, and the card stays in use.
		const cpu = await mockInvoke<LoadPlan>('plan_load', {
			modelId: 'glm-5.3-flash-ud-q2_k_xl',
			overrides: { vram_limit_mib: 12000 },
		});
		expect(cpu.gpu_layers).toBe(0);
		expect(cpu.args).not.toContain('--device');
		expect(cpu.tiers.vram_weights + cpu.tiers.vram_kv).toBe(0);
		expect(cpu.tiers.vram_expert_cache).toBeGreaterThan(0);
		expect(cpu.args[cpu.args.indexOf('-ngl') + 1]).toBe('0');
		for (const flag of ['--no-op-offload', '--no-kv-offload']) expect(cpu.args).toContain(flag);
		expect(cpu.args).not.toContain('--cpu-moe');
		expect(cpu.speed.decode_tps).toBeLessThanOrEqual(2.25);
		expect(cpu.native!.estimate_basis).toContain('2.25 tok/s');
		const noSsd = await mockInvoke<LoadPlan>('plan_load', {
			modelId: 'glm-5.3-flash-ud-q2_k_xl',
			overrides: { allow_ssd: false },
		});
		expect(noSsd.feasible).toBe(false);
	});

	it('plans Qwen3.6-35B-A3B fully in graphics memory once Kokoro is unloaded', async () => {
		const plan = await mockInvoke<LoadPlan>('plan_load', {
			modelId: 'qwen3.6-35b-a3b-ud-q4_k_m',
		});
		expect(plan.evict_alternative?.tier_mix).toBe('vram');
		expect(plan.evict_alternative?.label).toBe('fast');
		expect(plan.evict_alternative!.decode_tps).toBeGreaterThan(130);
		expect(plan.evict_alternative!.decode_tps).toBeLessThan(170);
		const outcome = await mockInvoke<ModelLoadOutcome>('load_model', {
			modelId: 'qwen3.6-35b-a3b-ud-q4_k_m',
		});
		expect(outcome.outcome).toBe('needs_eviction');
		await mockInvoke('orchestrator_unload', {
			residentId: 'pipeline:text_to_speech',
		});
		const freed = await mockInvoke<LoadPlan>('plan_load', {
			modelId: 'qwen3.6-35b-a3b-ud-q4_k_m',
		});
		expect(freed.tier_mix).toBe('vram');
		expect(freed.speed.label).toBe('fast');
		expect(freed.ctx).toBe(32768);
	});

	it('sizes GLM-5.3-Flash context like the planner and shrinks the expert cache as it grows', async () => {
		const MIB = 1024 * 1024;
		const rows = await mockInvoke<ContextMemory[]>('context_memory', {
			modelId: 'glm-5.3-flash-ud-q2_k_xl',
			contexts: [4096, 32768, 131072, 1048576],
		});
		// Same numbers as planner::context tests (MLA latent + DSA indexer, f16).
		expect(rows.map((r) => r.kv_bytes / MIB)).toEqual([44, 352, 1408, 11264]);
		expect(rows.map((r) => r.indexer_bytes / MIB)).toEqual([33, 264, 1056, 8448]);
		for (const r of rows) expect(r.state_bytes / MIB).toBeCloseTo(145.56, 1);
		const q = await mockInvoke<ContextMemory[]>('context_memory', {
			modelId: 'qwen3.6-35b-a3b-ud-q4_k_m',
			params: { ctx: 32768, kv_k: 'q8_0', kv_v: 'q4_0' },
		});
		// 10 full-attention layers × 512 K (q8_0) + 512 V (q4_0) per token.
		expect(q[0].kv_bytes).toBe(32768 * 10 * (16 * 34 + 16 * 18));

		const invoke = await emptyCard();
		const at = (ctx: number, vram_limit_mib?: number) =>
			invoke<LoadPlan>('plan_load', {
				modelId: 'glm-5.3-flash-ud-q2_k_xl',
				overrides: { ctx, kv_type: 'f16', vram_limit_mib },
			});
		// On the card a larger context takes its memory from the expert tier.
		let small = await at(8192);
		let big = await at(32768);
		expect(big.gpu_layers).toBe(-1);
		expect(big.context.kv_bytes).toBeGreaterThan(small.context.kv_bytes);
		expect(big.tiers.vram_kv).toBeGreaterThan(small.tiers.vram_kv);
		expect(big.native!.vram_budget_bytes).toBeLessThan(small.native!.vram_budget_bytes);
		expect(big.native!.expert_cache_mib).toBeLessThanOrEqual(small.native!.expert_cache_mib);
		expect(big.tiers.ssd).toBeGreaterThan(small.tiers.ssd);
		expect(small.args).toContain('-np');
		// A context that leaves too small a tier beside it moves the dense layers
		// back to the CPU, and from there it takes the system memory expert cache.
		const huge = await at(524288);
		expect([huge.feasible, huge.gpu_layers]).toEqual([true, 0]);
		small = await at(32768, 0);
		big = await at(262144, 0);
		expect(big.tiers.vram_kv).toBe(0);
		expect(big.native!.expert_cache_mib).toBeLessThan(small.native!.expert_cache_mib);
		expect(big.tiers.ssd).toBeGreaterThan(small.tiers.ssd);
	});

	it('caps the native tiers at the memory limits and streams the rest from the SSD', async () => {
		const MIB = 1024 * 1024;
		const GIB = 1024 * MIB;
		const invoke = await emptyCard();
		const at = (overrides: PlanOverrides) =>
			invoke<LoadPlan>('plan_load', {
				modelId: 'glm-5.3-flash-ud-q2_k_xl',
				overrides: { ctx: 8192, ...overrides },
			});
		const free = await at({});
		// A limit above what is free changes nothing: it never raises the budget.
		const roomy = await at({ ram_limit_mib: 512 * 1024, vram_limit_mib: 512 * 1024 });
		expect(roomy.tiers).toEqual(free.tiers);
		expect(roomy.native).toEqual(free.native);

		const lessRam = await at({ ram_limit_mib: 20 * 1024 });
		expect(lessRam.feasible).toBe(true);
		// The limit is RAM; the engine's envelope adds the dense layers and context
		// on the card, which Windows counts in the process.
		const card = (p: LoadPlan) =>
			p.tiers.vram_weights + p.tiers.vram_kv + p.tiers.vram_expert_cache + p.tiers.vram_overhead;
		const dense = card(lessRam) - lessRam.tiers.vram_expert_cache - 450 * MIB;
		expect(lessRam.gpu_layers).toBe(-1);
		expect(lessRam.native!.ram_budget_mib).toBe(Math.floor((20 * GIB + dense) / MIB));
		expect(lessRam.args).toContain(`engine.ram_budget_mib=${lessRam.native!.ram_budget_mib}`);
		expect(lessRam.tiers.ram).toBeLessThanOrEqual(20 * GIB);
		expect(lessRam.tiers.vram_expert_cache).toBe(free.tiers.vram_expert_cache);
		expect(lessRam.tiers.ssd).toBeGreaterThan(free.tiers.ssd);

		const cardLimit = Math.floor((card(free) * 0.75) / MIB);
		const lessVram = await at({ vram_limit_mib: cardLimit });
		const onCard = card(lessVram);
		expect(lessVram.tiers.vram_expert_cache).toBeGreaterThan(0);
		expect(onCard).toBeLessThanOrEqual(cardLimit * MIB);
		expect(lessVram.tiers.ssd).toBeGreaterThan(free.tiers.ssd);

		const offCard = await at({ vram_limit_mib: 0 });
		expect(offCard.feasible).toBe(true);
		expect(offCard.native!.vram_budget_bytes).toBe(0);
		expect(offCard.tiers.vram_overhead).toBe(0);
		expect(offCard.args).toContain('vram.budget=0');
		expect(offCard.args.join(' ')).not.toContain('vram.reserve_mib');
		// A CUDA build would still open a context on the card without this.
		expect(offCard.args[offCard.args.indexOf('--device') + 1]).toBe('none');
		for (const kept of [free, lessRam, lessVram]) expect(kept.args).not.toContain('--device');
		expect(offCard.warnings.some((w) => w.includes('graphics memory limit'))).toBe(true);

		const tiny = await at({ ram_limit_mib: 2 * 1024 });
		expect(tiny.feasible).toBe(false);
		expect(tiny.infeasible_reason).toContain('system memory limit');
	});

	it('caps llama-server tiers at the memory limits and keeps a model off the card at zero', async () => {
		const GIB = 1024 ** 3;
		const at = (overrides?: PlanOverrides) =>
			mockInvoke<LoadPlan>('plan_load', {
				modelId: 'llama-3.1-8b-instruct-q5_k_m',
				overrides: { ctx: 4096, ...overrides },
			});
		const vram = (p: LoadPlan) =>
			p.tiers.vram_weights + p.tiers.vram_expert_cache + p.tiers.vram_kv + p.tiers.vram_overhead;
		const free = await at();
		expect(free.engine_kind).toBe('llama_server');
		expect(vram(free)).toBeGreaterThan(0);
		expect(free.args).not.toContain('--device');
		expect((await at({ ram_limit_mib: 512 * 1024, vram_limit_mib: 512 * 1024 })).tiers).toEqual(
			free.tiers,
		);

		const offCard = await at({ vram_limit_mib: 0 });
		expect(offCard.feasible).toBe(true);
		expect(vram(offCard)).toBe(0);
		expect(offCard.tiers.ram).toBeGreaterThan(free.tiers.ram);
		const device = offCard.args.indexOf('--device');
		expect(offCard.args[device + 1]).toBe('none');
		expect(offCard.warnings.some((w) => w.includes('graphics memory limit'))).toBe(true);
		expect(offCard.warnings.some((w) => w.includes('No NVIDIA GPU'))).toBe(false);

		const partial = await at({ vram_limit_mib: 4 * 1024 });
		expect(vram(partial)).toBeGreaterThan(0);
		expect(vram(partial)).toBeLessThanOrEqual(4 * GIB);
		expect(vram(partial)).toBeLessThan(vram(free));
		expect(partial.args).not.toContain('--device');
	});

	it('refuses a context window larger than the system memory limit and keeps pinned experts on the card', async () => {
		const GIB = 1024 ** 3;
		const at = (modelId: string, overrides: PlanOverrides) =>
			mockInvoke<LoadPlan>('plan_load', { modelId, overrides });
		const vram = (p: LoadPlan) =>
			p.tiers.vram_weights + p.tiers.vram_expert_cache + p.tiers.vram_kv + p.tiers.vram_overhead;
		const llama = 'llama-3.1-8b-instruct-q5_k_m';

		// Off the card, the context window itself has to fit the system memory limit.
		const none = await at(llama, { ctx: 4096, ram_limit_mib: 0, vram_limit_mib: 0 });
		expect(none.engine_kind).toBe('llama_server');
		expect(none.feasible).toBe(false);
		expect(none.max_ctx).toBeNull();
		expect(none.infeasible_reason).toContain('0.0 GB system memory limit');
		expect(none.infeasible_reason).toContain('Raise the limit');
		expect(none.warnings.some((w) => w.includes('the system memory limit allows 0.0 GB'))).toBe(true);

		// A limit that only pushes weights to the SSD still loads.
		const spill = await at(llama, { ctx: 4096, ram_limit_mib: 2048, vram_limit_mib: 0 });
		expect(spill.feasible).toBe(true);
		expect(spill.tiers.ssd).toBeGreaterThan(0);
		expect(spill.tiers.ram).toBeLessThanOrEqual(2 * GIB);
		// With SSD streaming off the refusal names the limit that caused it.
		const noSsd = await at(llama, { ctx: 4096, ram_limit_mib: 2048, vram_limit_mib: 0, allow_ssd: false });
		expect(noSsd.feasible).toBe(false);
		expect(noSsd.infeasible_reason).toContain('system memory limit');

		// A pinned GPU expert placement is never moved off the card: it is refused.
		const moe = 'qwen3.6-35b-a3b-ud-q4_k_m';
		const pinned = await at(moe, { ctx: 4096, engine: 'llama_server', gpu_expert_layers: 999, vram_limit_mib: 0 });
		expect(pinned.engine_kind).toBe('llama_server');
		expect(pinned.feasible).toBe(false);
		expect(pinned.gpu_expert_layers).toBeGreaterThan(0);
		expect(vram(pinned)).toBeGreaterThan(0);
		expect(pinned.args).not.toContain('--device');
		expect(pinned.warnings.some((w) => w.includes('graphics memory limit allows'))).toBe(true);
		// Without the pin the same zero limit plans off the card.
		const unpinned = await at(moe, { ctx: 4096, engine: 'llama_server', vram_limit_mib: 0 });
		expect(vram(unpinned)).toBe(0);
		expect(unpinned.args[unpinned.args.indexOf('--device') + 1]).toBe('none');
	});

	it('keeps a MoE model partly on the card under a graphics limit below its always-used weights', async () => {
		const MIB = 1024 * 1024;
		const at = (overrides: PlanOverrides) =>
			mockInvoke<LoadPlan>('plan_load', {
				modelId: 'qwen3.6-35b-a3b-ud-q4_k_m',
				overrides: { ctx: 4096, engine: 'llama_server', ...overrides },
			});
		const vram = (p: LoadPlan) =>
			p.tiers.vram_weights + p.tiers.vram_expert_cache + p.tiers.vram_kv + p.tiers.vram_overhead;
		const free = await at({});
		expect(free.gpu_layers).toBe(-1);

		// Too small for every always-used weight, large enough to use the card:
		// fewer layers go on it and every expert stays on the host.
		const partial = await at({ vram_limit_mib: 3072 });
		expect(vram(partial)).toBeGreaterThan(0);
		expect(vram(partial)).toBeLessThanOrEqual(3072 * MIB);
		expect(partial.gpu_layers).toBeGreaterThanOrEqual(0);
		expect(partial.gpu_expert_layers).toBe(0);
		expect(partial.args).toContain('--cpu-moe');
		expect(partial.args).not.toContain('--device');
		expect(partial.tiers.vram_weights).toBeLessThan(free.tiers.vram_weights);
		expect(partial.warnings.some((w) => w.includes('graphics memory limit is below'))).toBe(false);
		expect(partial.warnings.some((w) => w.includes('layers fit in graphics memory'))).toBe(true);

		// Only a limit below the card's own overhead leaves the card out.
		const none = await at({ vram_limit_mib: Math.floor(partial.tiers.vram_overhead / MIB) - 1 });
		expect(vram(none)).toBe(0);
		expect(none.args[none.args.indexOf('--device') + 1]).toBe('none');
		const bare = await at({ vram_limit_mib: Math.ceil(partial.tiers.vram_overhead / MIB) });
		expect(bare.gpu_layers).toBe(0);
		expect(vram(bare)).toBe(bare.tiers.vram_overhead);
		expect(bare.args).not.toContain('--device');
	});

	it('names the system memory limit when llama-server is forced, and still refuses paging', async () => {
		const at = (overrides: PlanOverrides) =>
			mockInvoke<LoadPlan>('plan_load', {
				modelId: 'qwen3.6-35b-a3b-ud-q4_k_m',
				overrides: { ctx: 4096, engine: 'llama_server', ...overrides },
			});
		// The context cannot fit the user's own limit: changing engines would not help.
		const limited = await at({ ram_limit_mib: 0 });
		expect(limited.engine_kind).toBe('llama_server');
		expect(limited.feasible).toBe(false);
		expect(limited.infeasible_reason).toContain('0.0 GB system memory limit');
		expect(limited.infeasible_reason).not.toContain('overcommits');
		expect(limited.warnings.some((w) => w.includes('overcommits'))).toBe(false);
		// The context fits and only the weights spill: paging them is still refused.
		const paging = await at({ ram_limit_mib: 2048, vram_limit_mib: 6 * 1024 });
		expect(paging.tiers.ssd).toBeGreaterThan(0);
		expect(paging.feasible).toBe(false);
		expect(paging.infeasible_reason).toContain('overcommits RAM');
	});

	it('reports the largest context that fits when the requested one does not', async () => {
		const plan = await mockInvoke<LoadPlan>('plan_load', {
			modelId: 'glm-5.3-flash-ud-q2_k_xl',
			overrides: { ctx: 2_097_152, kv_type: 'f16' },
		});
		expect(plan.feasible).toBe(false);
		expect(plan.max_ctx).not.toBeNull();
		expect(plan.infeasible_reason).toContain(String(plan.max_ctx));
		const ok = await mockInvoke<LoadPlan>('plan_load', {
			modelId: 'glm-5.3-flash-ud-q2_k_xl',
			overrides: { ctx: plan.max_ctx!, kv_type: 'f16' },
		});
		expect(ok.feasible).toBe(true);
	});

	it('offers chat, not an agent workflow, for local text models', async () => {
		const models = await mockInvoke<InstalledModel[]>('list_local_models');
		const text = models.filter((model) => model.caps.runtime === 'llama-server');
		expect(text.length).toBeGreaterThan(0);
		for (const model of text) expect(model.caps.workflows).toEqual(['chat']);
	});

	it('refuses a stale dock unload once another model has replaced the text slot', async () => {
		vi.useFakeTimers();
		try {
			const models = await mockInvoke<InstalledModel[]>('list_local_models');
			const before = await mockInvoke<LedgerSnapshot>('orchestrator_ledger');
			const old = before.residents.find((r) => r.id === 'text');
			expect(old?.model_id).toBeTruthy();
			const next = models.find((m) => m.caps.runtime === 'llama-server' && m.id !== old?.model_id);
			expect(next).toBeTruthy();

			// B starts loading; the user hits A's stale x mid-replace.
			const load = mockInvoke('load_model', { modelId: next!.id, force: true });
			await vi.runAllTimersAsync();
			await load;
			await expect(
				mockInvoke('orchestrator_unload_if', { residentId: 'text', expectedModelId: old!.model_id,
				}),
			).rejects.toThrow(/Nothing was unloaded/);
			const after = await mockInvoke<LedgerSnapshot>('orchestrator_ledger');
			expect(after.residents.find((r) => r.id === 'text')?.model_id).toBe(next!.id);

			// The matching identity still unloads; an empty slot is a no-op.
			await mockInvoke('orchestrator_unload_if', { residentId: 'text', expectedModelId: next!.id,
			});
			await mockInvoke('orchestrator_unload_if', { residentId: 'text', expectedModelId: next!.id,
			});
			const empty = await mockInvoke<LedgerSnapshot>('orchestrator_ledger');
			expect(empty.residents.some((r) => r.id === 'text')).toBe(false);
		} finally {
			vi.useRealTimers();
		}
	});

	it('returns planner host footprints without a GPU', async () => {
		const store = new Map<string, string>([['oh.mock.gpu', 'none']]);
		vi.stubGlobal('localStorage', {
			getItem: (k: string) => store.get(k) ?? null,
			setItem: (k: string, v: string) => void store.set(k, v),
			removeItem: (k: string) => void store.delete(k),
		});
		try {
			const models = await mockInvoke<InstalledModel[]>('list_local_models');
			const m = models.find((x) => x.caps.runtime === 'llama-server')!;
			const fit = await mockInvoke<FitEstimate>('estimate_model_fit', { modelId: m.id, ctx: 8192,
			});
			expect(fit.gpu).toBe(false);
			expect(fit.plan).toBeTruthy();
			expect(fit.verdict).toBe('fits');
			const native = await mockInvoke<FitEstimate>('estimate_model_fit', { modelId: 'glm-5.3-flash-ud-q2_k_xl', ctx: 8192 });
			expect(native.gpu).toBe(false);
			expect(native.plan!.engine_kind).toBe('ssd_stream');
			expect(native.estimated_mb).toBe(Math.ceil(native.plan!.tiers.ram / 1024 ** 2));
			expect(native.plan!.tiers.vram_expert_cache).toBe(0);
			const gpu = await mockInvoke<{ kind: string }>('gpu_status');
			expect(gpu.kind).toBe('no_gpu');
		} finally {
			vi.unstubAllGlobals();
		}
	});

	it('reuses the streaming model only for a plain load, with its live plan', async () => {
		vi.useFakeTimers();
		try {
			const models = await mockInvoke<InstalledModel[]>('list_local_models');
			const llama = models.find((m) => m.display_name.startsWith('Llama 3.1 8B'))!;
			const load = mockInvoke<ModelLoadOutcome>('load_model', { modelId: llama.id, force: true,
			});
			await vi.runAllTimersAsync();
			const loaded = await load;
			expect(loaded.outcome).toBe('loaded');
			const livePlan = (loaded as { plan?: LoadPlan }).plan;
			expect(livePlan).toBeTruthy();

			// A reply is streaming from the loaded model.
			const conversation = await mockInvoke<ConversationSummaryDto>('create_conversation');
			await mockInvoke('send_message', { convId: conversation.id, content: 'Hi',
			});

			const reused = await mockInvoke<ModelLoadOutcome>('load_model', { modelId: llama.id,
			});
			expect(reused).toMatchObject({ outcome: 'loaded', model_id: llama.id, resident_id: 'text',
			});
			expect((reused as { plan?: LoadPlan }).plan).toEqual(livePlan);

			// Tuned, planned or forced requests would restart the server: refused.
			const busy = /chat reply is streaming/;
			await expect(
				mockInvoke('load_model', { modelId: llama.id, overrides: { ctx: 16384 },
				}),
			).rejects.toThrow(busy);
			await expect(
				mockInvoke('load_model', { modelId: llama.id, plan: livePlan })).rejects.toThrow(busy,
			);
			await expect(mockInvoke('load_model', { modelId: llama.id, force: true })).rejects.toThrow(
				busy,
			);

			await vi.runAllTimersAsync();
			await mockInvoke('delete_conversation', { id: conversation.id });
		} finally {
			vi.useRealTimers();
		}
	});
});
