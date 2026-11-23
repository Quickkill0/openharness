// prerender required by adapter-static to emit an index.html shell for the Tauri webview to load.
// ssr disabled because Tauri APIs require `window` (no Node/server context available).
export const ssr = false;
export const prerender = true;
