// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import FileTree from './FileTree.svelte';
import { mockInvoke } from '$lib/ipc/mock';
import { startDownload, startRepoDownload, listDownloads } from '$lib/ipc/commands';
import { downloadStore, clear } from '$lib/stores/downloads.svelte';
import { toasts } from '$lib/stores/toasts.svelte';
import type { HfFile, InstalledModel, QuantFit } from '$lib/types';

vi.mock('$lib/ipc/commands', () => ({
    startDownload: vi.fn(async () => 'single'),
    startRepoDownload: vi.fn(async () => 'group'),
    listDownloads: vi.fn(async () => []),
}));

const repoId = 'unsloth/GLM-5.3-Flash-GGUF';
async function fixture() {
    const files = await mockInvoke<HfFile[]>('hf_model_tree', { repoId });
    const quantFits = await mockInvoke<QuantFit[]>('store_quant_fits', { files });
    return { files, quantFits };
}

afterEach(() => { cleanup(); clear(); vi.clearAllMocks(); });

describe('split GGUF Store rows', () => {
    it('offers a complete set without GPU estimates and forwards the known HF task', async () => {
        const { files } = await fixture();
        render(FileTree, { files, repoId, quantFits: [], task: 'feature-extraction' });
        const table = screen.getByRole('table');
        expect(within(table).getAllByRole('row')).toHaveLength(2);
        const row = within(table).getByText(/split · 4 parts/).closest('tr')!;
        expect(row.textContent).toContain('100.6 GB');
        expect(row.textContent).toContain('No GPU estimate');
        await fireEvent.click(within(row).getByRole('button'));
        await waitFor(() => expect(startRepoDownload).toHaveBeenCalledWith({
            repoId, revision: 'main', format: 'gguf', task: 'feature-extraction',
            files: files.filter((f) => f.path.startsWith('UD-Q2_K_XL/')),
        }));
        expect(startDownload).not.toHaveBeenCalled();
    });
    it('shows one row with total size and four parts, and queues only the whole set', async () => {
        const { files, quantFits } = await fixture();
        const members = files.filter((f) => f.path.startsWith('UD-Q2_K_XL/'));
        const ondownloadqueued = vi.fn();
        vi.mocked(listDownloads).mockResolvedValueOnce(members.map((f, i) => ({
            id: `job-${i}`, group_id: 'group', repo_id: repoId, revision: 'main', file: f.path,
            dest_path: '', total_bytes: f.size, downloaded: 0, chunks: [], status: 'queued',
            created_at: 0, updated_at: 0,
        })));
        render(FileTree, { files, repoId, quantFits: [...quantFits].reverse(), ondownloadqueued });
        const table = screen.getByRole('table');
        // Header, split set, and unrelated mmproj file. No row per shard.
        expect(within(table).getAllByRole('row')).toHaveLength(3);
        const tag = within(table).getByText(/split · 4 parts/);
        expect(tag.closest('tr')!.textContent).toContain('100.6 GB');
        const download = within(tag.closest('tr')!).getByRole('button', { name: 'Download Q2_K_XL' });
        await fireEvent.click(download);
        await waitFor(() => expect(startRepoDownload).toHaveBeenCalledWith({
            repoId, revision: 'main', files: members, format: 'gguf', task: null,
        }));
        await waitFor(() => expect(ondownloadqueued).toHaveBeenCalledOnce());
        expect(startDownload).not.toHaveBeenCalled();
        expect(downloadStore.jobs.size).toBe(4);
        expect([...downloadStore.jobs.values()].every((j) => j.group_id === 'group')).toBe(true);
        await fireEvent.click(screen.getByRole('button', { name: /All files & formats/ }));
        for (const file of members) expect(screen.queryByRole('button', { name: `Download ${file.path.split('/').pop()}` })).toBeNull();
    });

    it('shows the backend installed refusal without queuing a partial set', async () => {
        const { files } = await fixture();
        const message = 'This model is already installed. Remove it from Models before downloading it again.';
        // AppError serializes as a string at the Tauri boundary.
        vi.mocked(startRepoDownload).mockRejectedValueOnce(message);
        const ondownloadqueued = vi.fn();
        render(FileTree, { files, repoId, quantFits: [], installed: [], ondownloadqueued });
        const row = screen.getByText(/split · 4 parts/).closest('tr')!;
        await fireEvent.click(within(row).getByRole('button'));
        await waitFor(() => expect(toasts.value.some((t) => t.variant === 'error'
            && t.message === `Download failed: ${message}`)).toBe(true));
        expect(listDownloads).not.toHaveBeenCalled();
        expect(ondownloadqueued).not.toHaveBeenCalled();
        expect(downloadStore.jobs.size).toBe(0);
        expect(within(row).getByRole('button').hasAttribute('disabled')).toBe(false);
    });

    it('marks only the exact complete set installed, not another folder or a partial install', async () => {
        const { files, quantFits } = await fixture();
        const models = await mockInvoke<InstalledModel[]>('list_local_models');
        const glm = models.find((m) => m.repo_id === repoId)!;
        const primary = files.find((f) => f.path.includes('-00001-of-'))!;
        const model = { ...glm, primary_rel_path: primary.path };
        const view = render(FileTree, { files, repoId, quantFits, installed: [model] });
        expect(within(screen.getByRole('table')).getAllByText('Installed')).toHaveLength(1);
        await view.rerender({ files, repoId, quantFits, installed: [{ ...model, file_count: 3 }] });
        expect(within(screen.getByRole('table')).queryByText('Installed')).toBeNull();
        await view.rerender({ files, repoId, quantFits, installed: [{ ...model, primary_rel_path: primary.path.replace('UD-Q2_K_XL/', 'other/') }] });
        expect(within(screen.getByRole('table')).queryByText('Installed')).toBeNull();
    });
});
