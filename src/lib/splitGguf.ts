import type { HfFile } from '$lib/types';

/** The repo-relative stem includes the folder, not just the quant label. */
export function splitGgufPart(path: string) {
    const m = path.match(/^(.*)-(\d+)-of-(\d+)\.gguf$/);
    if (!m || Number(m[3]) <= 1) return null;
    return { key: m[1], number: Number(m[2]), count: Number(m[3]) };
}

export function splitGgufFiles(file: HfFile, files: HfFile[]): HfFile[] {
    const part = splitGgufPart(file.path);
    if (!part) return [file];
    const members = files.filter((f) => splitGgufPart(f.path)?.key === part.key)
        .sort((a, b) => a.path.localeCompare(b.path));
    if (members.length !== part.count || members.some((f, i) => {
        const p = splitGgufPart(f.path)!;
        return p.count !== part.count || p.number !== i + 1;
    })) throw new Error('Download one complete split GGUF set, including every part.');
    return members;
}
