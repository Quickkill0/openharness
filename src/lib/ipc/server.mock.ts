import type { ServerSettings, ServerStatus, ServerRequestLog } from './server';

let key = 'oh-mock-key-for-browser-preview-only';
let keyVersion = 0;
let settings: ServerSettings = {
    enabled: true, port: 1234, bind: 'home_network', require_key: true,
    auto_load: true, idle_minutes: 15, aliases: { 'qwen2.5-coder-7b-instruct-q4_k_m': 'qwen-coder' },
};
const models = [{ id: 'qwen2.5-coder-7b-instruct-q4_k_m', name: 'Qwen 2.5 Coder 7B', alias: 'qwen-coder', loaded: true }];
const logs: ServerRequestLog[] = [{
    time: new Date().toISOString(), client_ip: '192.168.1.20', model: models[0].id,
    prompt_tokens: 1240, completion_tokens: 186, tokens_per_second: 62.4, status: 200, duration_ms: 2981,
}];
const listeners = new Set<(entry: ServerRequestLog) => void>();
let timer: ReturnType<typeof setInterval> | undefined;
export function mockServerSubscribe(onmessage: (entry: ServerRequestLog) => void) {
    listeners.add(onmessage);
    if (typeof window !== 'undefined' && !timer) timer = setInterval(() => {
        if (!settings.enabled) return;
        const entry = { ...logs[0], time: new Date().toISOString() };
        logs.push(entry);
        if (logs.length > 500) logs.shift();
        for (const listener of listeners) listener(structuredClone(entry));
    }, 10000);
    return { onmessage };
}
function status(): ServerStatus {
    return structuredClone({ running: settings.enabled, settings, key_configured: true, error: null,
        models: models.map(m => ({ ...m, alias: settings.aliases[m.id] ?? 'qwen-2-5-coder-7b' })) });
}
export async function mockServerInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
    switch (cmd) {
        case 'server_status': return status() as T;
        case 'server_start': settings.enabled = true; return status() as T;
        case 'server_stop': settings.enabled = false; return status() as T;
        case 'server_update_settings': {
            const next = structuredClone(args?.settings as ServerSettings);
            if (!Number.isInteger(next.port) || next.port < 1 || next.port > 65535) throw new Error('Use a port from 1 to 65535');
            if (!Number.isInteger(next.idle_minutes) || next.idle_minutes < 0 || next.idle_minutes > 525600) throw new Error('Use an idle timeout from 0 to 525600 minutes');
            const aliases = Object.values(next.aliases);
            if (aliases.some(alias => !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(alias) || alias.length > 128) || new Set(aliases).size !== aliases.length) throw new Error('Aliases must be unique lowercase model slugs');
            settings = next;
            return status() as T;
        }
        case 'server_regenerate_key': key = `oh-mock-preview-key-${++keyVersion}`; return undefined as T;
        case 'server_get_key': return key as T;
        case 'server_request_log': return structuredClone(logs) as T;
        case 'server_connection_info': return {
            this_pc: `http://127.0.0.1:${settings.port}/v1`,
            lan: settings.bind === 'home_network' ? [`http://192.168.1.42:${settings.port}/v1`] : [],
            auth_required: settings.bind === 'home_network' || settings.require_key,
        } as T;
        default: throw new Error(`Unknown server command: ${cmd}`);
    }
}
