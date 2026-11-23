import { afterEach, beforeEach, expect, it, vi } from 'vitest';
let mockInvoke: typeof import('./mock').mockInvoke;
let mockSubscribe: typeof import('./mock').mockSubscribe;
beforeEach(async () => {
    vi.resetModules();
    ({ mockInvoke, mockSubscribe } = await import('./mock'));
});
import { splitGgufFiles } from '$lib/splitGguf';
import type { DownloadProgress, HfFile, InstalledModel, Job, QuantFit } from '$lib/types';

afterEach(() => vi.useRealTimers());

it('mock split downloads reject single parts and publish one complete file-backed model', async () => {
    vi.useFakeTimers();
    const repoId = 'unsloth/GLM-5.3-Flash-GGUF';
    const files = await mockInvoke<HfFile[]>('hf_model_tree', { repoId });
    const primary = files.find((f) => f.path.includes('-00001-of-'))!;
    const members = splitGgufFiles(primary, files);
    const fits = await mockInvoke<QuantFit[]>('store_quant_fits', { files });
    const fit = fits.find((f) => f.path === primary.path)!;
    expect(fit.set_key).toBe('UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL');
    expect(fit.part_count).toBe(4);
    expect(fit.set_size).toBe(members.reduce((s, f) => s + f.size, 0));
    expect(fit.best).toBe(false);
    await expect(mockInvoke('start_download', { repoId, revision: 'main', file: primary })).rejects.toThrow('complete set');
    await expect(mockInvoke('start_repo_download', { repoId, revision: 'test', files: [primary], format: 'gguf' })).rejects.toThrow();
    const groupId = await mockInvoke<string>('start_repo_download', { repoId, revision: 'test', files: members, format: 'gguf', task: null });
    const jobs = await mockInvoke<Job[]>('list_downloads');
    expect(jobs.filter((j) => j.group_id === groupId).map((j) => j.file)).toEqual(members.map((f) => f.path));
    await vi.runAllTimersAsync();
    const models = (await mockInvoke<InstalledModel[]>('list_local_models')).filter((m) => m.repo_id === repoId && m.revision === 'test');
    expect(models).toHaveLength(1);
    expect(models[0].primary_rel_path).toBe(primary.path);
    expect(models[0].primary_file_path).toContain(primary.path.replaceAll('/', '\\'));
    expect(models[0].local_path).toBeNull();
    expect(models[0].file_count).toBe(4);
    expect(models[0].total_size).toBe(fit.set_size);
    expect(models[0].task).toBeNull();
    expect(models[0].capability).toBe('other');
    expect(models[0].caps.workflows).toContain('chat');
});

it('mock mirrors installed refusal, alias closure, duplicate rejection and superseded retry', async () => {
    vi.useFakeTimers();
    const events: DownloadProgress[] = [];
    mockSubscribe<DownloadProgress>('subscribe_download_progress', (event) => events.push(event));
    const fixture = await mockInvoke<HfFile[]>('hf_model_tree', { repoId: 'unsloth/GLM-5.3-Flash-GGUF' });
    const files = splitGgufFiles(fixture.find((f) => f.path.includes('-00001-of-'))!, fixture);
    const repoId = 'safety/É-repo';
    const request = { repoId, revision: 'C', files, format: 'gguf', task: null };
    const c = await mockInvoke<string>('start_repo_download', request);
    const aliases = files.map((f) => ({ ...f, path: f.path.toLowerCase().replaceAll('/', '\\') }));
    const replacement = { ...request, repoId: repoId.toLowerCase(), revision: 'A', files: aliases };
    await expect(mockInvoke('start_repo_download', replacement)).rejects.toThrow('already have a download');
    await mockInvoke('cancel_download_group', { id: c });
    const a = await mockInvoke<string>('start_repo_download', replacement);
    await vi.runAllTimersAsync();
    const installed = (await mockInvoke<InstalledModel[]>('list_local_models')).find((m) => m.repo_id === replacement.repoId)!;
    expect(installed.file_count).toBe(4);
    const count = (await mockInvoke<Job[]>('list_downloads')).length;
    await expect(mockInvoke('start_repo_download', { ...request, revision: 'B' })).rejects.toThrow('Remove it from Models');
    expect((await mockInvoke<Job[]>('list_downloads')).length).toBe(count);
    await expect(mockInvoke('start_repo_download', { ...request, files: [...files, aliases[0]] })).rejects.toThrow('same file');
    // Persisted resumable history cannot regain rights after removing A's history.
    await mockInvoke('delete_download_group', { id: a });
    const old = (await mockInvoke<Job[]>('list_downloads')).filter((j) => j.group_id === c);
    for (const job of old) { job.status = 'failed'; job.error = 'old integrity error'; }
    await mockInvoke('resume_download_group', { id: c });
    expect(old.every((j) => j.status === 'failed' && j.error?.includes('superseded'))).toBe(true);
    for (const job of old) expect(events.filter((e) => e.id === job.id).at(-1)?.error).toContain('superseded');
    await mockInvoke('delete_local_model', { id: installed.id });
    const fresh = await mockInvoke<string>('start_repo_download', { ...request, revision: 'fresh' });
    const retry = (await mockInvoke<Job[]>('list_downloads')).find((j) => j.group_id === fresh)!;
    retry.status = 'failed'; retry.error = 'old network error';
    await mockInvoke('resume_download', { id: retry.id });
    expect(events.filter((e) => e.id === retry.id).at(-1)).toMatchObject({ status: 'queued', error: null });
    await vi.runAllTimersAsync();
    expect((await mockInvoke<Job[]>('list_downloads')).filter((j) => j.group_id === fresh).every((j) => j.status === 'done')).toBe(true);
    expect((await mockInvoke<InstalledModel[]>('list_local_models')).some((m) => m.repo_id === repoId && m.revision === 'fresh')).toBe(true);
});

it.each([
    ['text-generation', 'text-generation', ['chat'], null],
    ['feature-extraction', 'embedding', ['embeddings'], null],
    ['sentence-similarity', 'embedding', ['embeddings'], null],
    ['automatic-speech-recognition', 'speech', [], 'GGUF speech'],
    ['text-to-speech', 'speech', [], 'GGUF speech'],
    ['audio-to-audio', 'speech', [], 'GGUF speech'],
    ['text-to-image', 'image-gen', [], 'GGUF diffusion'],
    ['image-to-image', 'image-gen', [], 'GGUF diffusion'],
    ['text-to-video', 'image-gen', [], 'GGUF diffusion'],
    ['image-to-video', 'image-gen', [], 'GGUF diffusion'],
    ['image-text-to-text', 'vision-text', ['chat'], null],
    ['visual-question-answering', 'vision-text', ['chat'], null],
    ['unknown-task', 'other', ['chat'], null],
] as const)('completed split task %s matches backend capability and workflows', async (task, capability, workflows, reason) => {
    vi.useFakeTimers();
    const repoId = 'unsloth/GLM-5.3-Flash-GGUF';
    const files = await mockInvoke<HfFile[]>('hf_model_tree', { repoId });
    const members = splitGgufFiles(files.find((f) => f.path.includes('-00001-of-'))!, files);
    const revision = `task-parity-${task}`;
    await mockInvoke('start_repo_download', { repoId, revision, files: members, format: 'gguf', task });
    await vi.runAllTimersAsync();
    const model = (await mockInvoke<InstalledModel[]>('list_local_models'))
        .find((m) => m.repo_id === repoId && m.revision === revision)!;
    expect(model.task).toBe(task);
    expect(model.capability).toBe(capability);
    expect(model.caps.workflows).toEqual(workflows);
    expect(model.caps.runtime).toBe(reason ? null : 'llama-server');
    if (reason) expect(model.caps.reason).toContain(reason);
    else expect(model.caps.reason).toBeNull();
});
