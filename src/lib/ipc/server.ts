import { invoke, subscribe } from './backend';

export interface ServerSettings {
    enabled: boolean;
    port: number;
    bind: 'this_pc' | 'home_network';
    require_key: boolean;
    auto_load: boolean;
    idle_minutes: number;
    aliases: Record<string, string>;
}
export interface ApiModel { id: string; name: string; alias: string; loaded: boolean }
export interface ServerStatus {
    running: boolean;
    settings: ServerSettings;
    key_configured: boolean;
    error: string | null;
    models: ApiModel[];
}
export interface ServerConnectionInfo { this_pc: string; lan: string[]; auth_required: boolean }
export interface ServerRequestLog {
    time: string;
    client_ip: string;
    model: string | null;
    prompt_tokens: number | null;
    completion_tokens: number | null;
    tokens_per_second: number | null;
    status: number;
    duration_ms: number;
}
export const serverStatus = () => invoke<ServerStatus>('server_status');
export const serverStart = () => invoke<ServerStatus>('server_start');
export const serverStop = () => invoke<ServerStatus>('server_stop');
export const serverUpdateSettings = (settings: ServerSettings) => invoke<ServerStatus>('server_update_settings', { settings });
export const serverRegenerateKey = () => invoke<void>('server_regenerate_key');
export const serverGetKey = () => invoke<string | null>('server_get_key');
export const serverRequestLog = () => invoke<ServerRequestLog[]>('server_request_log');
export const serverConnectionInfo = () => invoke<ServerConnectionInfo>('server_connection_info');
export const serverSubscribeRequestLog = (onmessage: (entry: ServerRequestLog) => void) => subscribe('server_subscribe_request_log', onmessage);
