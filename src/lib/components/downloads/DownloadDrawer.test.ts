// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import DownloadDrawer from './DownloadDrawer.svelte';
import { listDownloads, resumeDownloadGroup } from '$lib/ipc/commands';
import { subscribeDownloadProgress } from '$lib/ipc/channels';
import { clear, downloadStore } from '$lib/stores/downloads.svelte';
import type { DownloadProgress, Job } from '$lib/types';

vi.mock('$lib/ipc/commands', () => ({
    listDownloads: vi.fn(),
    pauseDownload: vi.fn(), resumeDownload: vi.fn(), cancelDownload: vi.fn(), deleteDownload: vi.fn(),
    pauseDownloadGroup: vi.fn(), resumeDownloadGroup: vi.fn(), cancelDownloadGroup: vi.fn(), deleteDownloadGroup: vi.fn(),
}));
vi.mock('$lib/ipc/channels', () => ({ subscribeDownloadProgress: vi.fn() }));
afterEach(() => { cleanup(); clear(); vi.clearAllMocks(); });

it('renders current refusal and ordinary failure messages from events without refetching', async () => {
    const makeJob = (id: string, group: string | null, error?: string): Job => ({
        id, group_id: group, repo_id: 'owner/model', revision: 'main', file: `${id}.gguf`,
        dest_path: '', status: error ? 'failed' : 'paused', error, downloaded: 10, total_bytes: 20,
        chunks: [], created_at: 0, updated_at: 0,
    });
    const jobs = [makeJob('first', 'set', 'old integrity error'), makeJob('second', 'set'), makeJob('single', null)];
    vi.mocked(listDownloads).mockResolvedValueOnce(jobs);
    let progress!: (event: DownloadProgress) => void;
    vi.mocked(subscribeDownloadProgress).mockImplementation(async (callback) => {
        progress = callback;
        return { onmessage: callback };
    });
    const event = (id: string, status: DownloadProgress['status'], error: string | null): DownloadProgress => ({
        id, status, error, downloaded: 10, total: 20, bytes_per_sec: 0, phase: 'downloading',
    });
    const reason = 'This download was superseded. Start a new download from the Store.';
    vi.mocked(resumeDownloadGroup).mockImplementationOnce(async () => {
        progress(event('first', 'failed', reason));
        progress(event('second', 'failed', reason));
    });
    render(DownloadDrawer, { open: true });
    await waitFor(() => expect(subscribeDownloadProgress).toHaveBeenCalledOnce());
    expect(screen.getByText(/old integrity error/)).toBeTruthy();
    await fireEvent.click(within(screen.getByText(/Whole repo,/).closest('.wb-dl')!).getByRole('button', { name: 'Resume' }));
    await waitFor(() => expect(screen.getAllByText(new RegExp(reason.replace('.', '\\.')))).toHaveLength(2));
    expect(screen.queryByText(/old integrity error/)).toBeNull();
    expect(downloadStore.jobs.get('first')?.error).toBe(reason);
    // Ordinary failures use the same field. A storm adds no listDownloads calls.
    for (let i = 0; i < 300; i++) progress(event('single', 'failed', 'Network disconnected'));
    await waitFor(() => expect(screen.getByText('Network disconnected')).toBeTruthy());
    expect(listDownloads).toHaveBeenCalledOnce();
    progress(event('single', 'queued', null));
    await waitFor(() => expect(screen.queryByText('Network disconnected')).toBeNull());
    expect(downloadStore.jobs.get('single')?.error).toBeUndefined();
});
