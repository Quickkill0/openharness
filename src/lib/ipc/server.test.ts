import { describe, it, expect, vi } from 'vitest';
import { mockInvoke, mockSubscribe } from './mock';
import type { ServerStatus, ServerConnectionInfo, ServerRequestLog } from './server';

describe('API server mock IPC', () => {
    it('can regenerate preview keys without secure-context browser crypto APIs', async () => {
        vi.stubGlobal('crypto', undefined);
        try {
            const before = await mockInvoke('server_get_key');
            await mockInvoke('server_regenerate_key');
            expect(await mockInvoke('server_get_key')).not.toBe(before);
        } finally {
            vi.unstubAllGlobals();
        }
    });
    it('streams metadata-only requests from the mock client and stops streaming when disabled', async () => {
        vi.useFakeTimers();
        vi.stubGlobal('window', {});
        try {
            const received: ServerRequestLog[] = [];
            mockSubscribe<ServerRequestLog>('server_subscribe_request_log', entry => received.push(entry));
            await vi.advanceTimersByTimeAsync(10000);
            expect(received).toHaveLength(1);
            expect(received[0].client_ip).toBe('192.168.1.20');
            await mockInvoke('server_stop');
            await vi.advanceTimersByTimeAsync(10000);
            expect(received).toHaveLength(1);
            await mockInvoke('server_start');
        } finally {
            vi.clearAllTimers();
            vi.useRealTimers();
            vi.unstubAllGlobals();
        }
    });
    it('reports a running LAN server and the AI Shuffle client without leaking a key', async () => {
        const status = await mockInvoke<ServerStatus>('server_status');
        expect(status.running).toBe(true);
        expect(status.settings.bind).toBe('home_network');
        expect(JSON.stringify(status)).not.toContain('oh-mock');
        const connection = await mockInvoke<ServerConnectionInfo>('server_connection_info');
        expect(connection.this_pc).toBe('http://127.0.0.1:1234/v1');
        expect(connection.lan).toEqual(['http://192.168.1.42:1234/v1']);
        expect(connection.auth_required).toBe(true);
        const logs = await mockInvoke<ServerRequestLog[]>('server_request_log');
        expect(logs[0].client_ip).toBe('192.168.1.20');
        expect(JSON.stringify(logs)).not.toMatch(/messages|prompt"|content/);
        expect(mockSubscribe<ServerRequestLog>('server_subscribe_request_log', () => {}).onmessage).toBeTypeOf('function');
    });
    it('supports settings, start/stop, aliases and explicit key retrieval/rotation', async () => {
        expect((await mockInvoke<ServerStatus>('server_stop')).running).toBe(false);
        const before = await mockInvoke<string>('server_get_key');
        await mockInvoke('server_regenerate_key');
        expect(await mockInvoke('server_get_key')).not.toBe(before);
        const status = await mockInvoke<ServerStatus>('server_status');
        status.settings.port = 1235;
        status.settings.bind = 'this_pc';
        status.settings.require_key = false;
        status.settings.aliases[status.models[0].id] = 'my-model';
        await mockInvoke('server_update_settings', { settings: status.settings });
        const next = await mockInvoke<ServerStatus>('server_start');
        expect(next.running).toBe(true);
        expect(next.models[0].alias).toBe('my-model');
        const info = await mockInvoke<ServerConnectionInfo>('server_connection_info');
        expect(info.lan).toEqual([]);
        expect(info.auth_required).toBe(false);
        expect(info.this_pc).toBe('http://127.0.0.1:1235/v1');
        await expect(mockInvoke('server_update_settings', { settings: { ...next.settings, port: 0 } })).rejects.toThrow('port');
        await expect(mockInvoke('server_update_settings', { settings: { ...next.settings, aliases: { a: 'Same Alias' } } })).rejects.toThrow('Aliases');
    });
});
