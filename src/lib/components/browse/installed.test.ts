import { describe, expect, it } from 'vitest';
import { isFileInstalled, isRepoInstallComplete } from './installed';

const SHA_A = 'a'.repeat(64);
const SHA_B = 'b'.repeat(64);

/**
 * A single GGUF as the backend registers it: hash id, stored on disk as
 * `<sha256>.gguf`, original repo filename kept in `primary_rel_path`.
 */
const ggufRow = {
	id: SHA_A,
	primary_rel_path: 'A-Q4_K_M.gguf',
	primary_sha256: SHA_A,
	local_path: null,
	file_count: 1,
};

describe('isFileInstalled', () => {
	it('matches the HF hash against the catalog row', () => {
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf', sha256: SHA_A }, [ggufRow])).toBe(true);
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf', sha256: SHA_A.toUpperCase() }, [ggufRow])).toBe(true);
		// The id alone is enough when the primary hash was not recorded.
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf', sha256: SHA_A }, [{ ...ggufRow, primary_sha256: null }])).toBe(
			true,
		);
	});

	it('does not mark a different file with the same quant label as installed', () => {
		// Review repro: A-Q4_K_M installed, B-Q4_K_M must keep its download button.
		expect(isFileInstalled({ path: 'B-Q4_K_M.gguf', sha256: SHA_B }, [ggufRow])).toBe(false);
		expect(isFileInstalled({ path: 'B-Q4_K_M.gguf' }, [ggufRow])).toBe(false);
	});

	it('uses only the hash when one is given, even if the name matches', () => {
		// Same name, different content (re-uploaded file): not installed.
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf', sha256: SHA_B }, [ggufRow])).toBe(false);
	});

	it('never matches on the content-addressed name on disk', () => {
		// A repo file literally named `<installed-sha>.gguf` but with its own hash.
		expect(isFileInstalled({ path: `${SHA_A}.gguf`, sha256: SHA_B }, [ggufRow])).toBe(false);
		expect(isFileInstalled({ path: `${SHA_A}.gguf` }, [ggufRow])).toBe(false);
	});

	it('falls back to the original repo filename when the file has no hash', () => {
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf' }, [ggufRow])).toBe(true);
		expect(isFileInstalled({ path: 'sub/a-q4_k_m.gguf', sha256: null }, [ggufRow])).toBe(true);
		// A row without a recorded repo filename cannot match by name.
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf' }, [{ ...ggufRow, primary_rel_path: null }])).toBe(false);
	});

	it('is false with nothing installed', () => {
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf', sha256: SHA_A }, [])).toBe(false);
		expect(isFileInstalled({ path: 'A-Q4_K_M.gguf' }, [])).toBe(false);
	});
});

describe('isRepoInstallComplete', () => {
	const folder = (file_count: number) => ({
		id: 'f',
		primary_rel_path: 'model.safetensors',
		primary_sha256: null,
		local_path: 'C:\\OH\\models\\repo',
		file_count,
	});

	it('needs a folder install with every file "Download repo" fetches', () => {
		expect(isRepoInstallComplete([folder(12)], 12)).toBe(true);
		expect(isRepoInstallComplete([folder(14)], 12)).toBe(true);
	});

	it('does not count a partial folder or a lone file from the repo', () => {
		// Review repro: one file from a dir-layout repo must not read as Installed.
		expect(isRepoInstallComplete([folder(1)], 12)).toBe(false);
		expect(isRepoInstallComplete([ggufRow], 1)).toBe(false);
	});

	it('is false for an empty download set or no rows', () => {
		expect(isRepoInstallComplete([folder(3)], 0)).toBe(false);
		expect(isRepoInstallComplete([], 3)).toBe(false);
	});
});
