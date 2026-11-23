/**
 * Shared markdown pipeline + streaming-markdown helpers.
 *
 * One parser (marked), one sanitizer (DOMPurify, single config), one CSS scope
 * (`.chat-md`, see `components/chat/chat-md.css`) — used by both the static
 * `MarkdownMessage` and the live `StreamingMarkdown`, so a streamed turn looks
 * *identical* to its persisted render and there is no end-of-turn restyle.
 *
 * The streaming helpers are pure functions (unit-tested in `markdown.test.ts`):
 *
 * - `findStableBoundary(text)` — index of the last block boundary (blank line
 *   outside any code fence, with split-safe neighbours) so callers can cache
 *   the parsed prefix and re-parse only the small active tail per update.
 * - `repairTail(tail)` — closes incomplete constructs (open ``` fence, inline
 *   code, **bold**, *italic*, __/_/~~) so partial markdown renders in its
 *   final style from the first token, and holds back constructs that cannot
 *   be stabilized (partial table header, dangling `[link`) as plain text
 *   until they settle.
 */

export type RenderMarkdown = (md: string) => string;

const DOMPURIFY_CONFIG = {
	ALLOWED_TAGS: [
		'p', 'br', 'strong', 'em', 'del', 'code', 'pre', 'blockquote',
		'ul', 'ol', 'li', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
		'table', 'thead', 'tbody', 'tr', 'th', 'td', 'hr', 'span',
	],
	ALLOWED_ATTR: ['href', 'title', 'class'],
	ALLOW_DATA_ATTR: false,
};

function escapeHtml(text: string): string {
	return text.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]!);
}

let rendererPromise: Promise<RenderMarkdown> | null = null;

/**
 * Lazy-loads marked + DOMPurify once (module-level singleton — the link
 * `target=_blank` hook is registered exactly once, instead of once per
 * component instance). Falls back to an escaped `<pre>` renderer if the
 * imports fail.
 */
export function loadMarkdownRenderer(): Promise<RenderMarkdown> {
	if (!rendererPromise) {
		rendererPromise = (async () => {
			try {
				const [{ marked }, { default: purify }] = await Promise.all([
					import('marked'),
					import('dompurify'),
				]);
				purify.addHook('afterSanitizeAttributes', (node: Element) => {
					if (node.tagName === 'A') {
						node.setAttribute('target', '_blank');
						node.setAttribute('rel', 'noopener noreferrer nofollow');
					}
				});
				return (md: string) =>
					purify.sanitize(marked(md) as string, DOMPURIFY_CONFIG) as string;
			} catch {
				return (md: string) => `<pre>${escapeHtml(md)}</pre>`;
			}
		})();
	}
	return rendererPromise;
}

/** Decorates fenced code blocks under `root` with a copy button (idempotent). */
export function decorateCodeBlocks(root: HTMLElement): void {
	root.querySelectorAll('pre').forEach((pre) => {
		if (pre.querySelector('.code-copy')) return;
		const btn = document.createElement('button');
		btn.type = 'button';
		btn.className = 'code-copy';
		btn.textContent = 'Copy';
		btn.addEventListener('click', () => {
			const code = pre.querySelector('code')?.textContent ?? pre.textContent ?? '';
			navigator.clipboard?.writeText(code);
			btn.textContent = 'Copied';
			setTimeout(() => (btn.textContent = 'Copy'), 1200);
		});
		pre.appendChild(btn);
	});
}

// ---------------------------------------------------------------------------
// Streaming helpers (pure)
// ---------------------------------------------------------------------------

interface Line {
	start: number;
	text: string;
}

function splitLines(text: string): Line[] {
	const lines: Line[] = [];
	let start = 0;
	for (;;) {
		const nl = text.indexOf('\n', start);
		if (nl === -1) {
			lines.push({ start, text: text.slice(start) });
			return lines;
		}
		lines.push({ start, text: text.slice(start, nl) });
		start = nl + 1;
	}
}

const FENCE_RE = /^ {0,3}(`{3,}|~{3,})/;

/** Fence-state tracker: returns the new open-fence marker (or null). */
function fenceStep(line: string, open: string | null): string | null {
	const m = FENCE_RE.exec(line);
	if (!m) return open;
	const marker = m[1];
	if (open === null) return marker;
	// A closing fence uses the same character, at least as long.
	if (marker[0] === open[0] && marker.length >= open.length) return null;
	return open;
}

const LIST_ITEM_RE = /^ {0,3}(?:[-*+]|\d{1,9}[.)])(?: |$)/;
const INDENTED_RE = /^(?: {4}|\t)/;
const BLANK_RE = /^\s*$/;
const HEADING_RE = /^ {0,3}#{1,6}(?: |$)/;

/**
 * Can this line merge across a blank line with a like neighbour? Splitting
 * between two list items turns one loose list into two tight lists; splitting
 * inside an indented code block turns one block into two.
 */
function continuable(line: string): boolean {
	return LIST_ITEM_RE.test(line) || INDENTED_RE.test(line);
}

/**
 * Index of the last stable block boundary in `text`: the start of the line
 * following a blank line that sits outside any code fence and whose nearest
 * non-blank neighbours are both safe to split between. Returns 0 when no such
 * boundary exists yet. `text.slice(0, boundary)` parses identically alone or
 * as part of the whole document, so its HTML can be cached.
 */
export function findStableBoundary(text: string): number {
	const lines = splitLines(text);
	let fence: string | null = null;
	let best = 0;
	let prevNonBlank: string | null = null;
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i].text;
		if (fence === null && BLANK_RE.test(line)) {
			// Find the next non-blank line; without one the boundary is not
			// stable yet (the next block hasn't started streaming).
			let j = i + 1;
			while (j < lines.length && BLANK_RE.test(lines[j].text)) j++;
			if (j < lines.length && prevNonBlank !== null) {
				const next = lines[j].text;
				if (!(continuable(prevNonBlank) && continuable(next))) {
					best = lines[i + 1]?.start ?? best;
				}
			}
			continue;
		}
		fence = fenceStep(line, fence);
		if (!BLANK_RE.test(line)) prevNonBlank = line;
	}
	return best;
}

export interface RepairedTail {
	/** Markdown safe to parse — incomplete constructs closed. */
	md: string;
	/** Dangling inline fragment (e.g. `[link` mid-typing) to show as plain text at the cursor. */
	inlinePlain: string;
	/** Held-back trailing lines (e.g. a table header before its delimiter row) shown as a plain block. */
	blockPlain: string;
}

/** True when `line` looks like a (possibly partial) GFM table delimiter row. */
function isDelimiterRow(line: string): boolean {
	return /^ {0,3}\|[ \t|:-]*$/.test(line) && line.includes('-');
}

/**
 * Repairs the streaming tail so it renders in its final style:
 * fence/inline-code/emphasis closed, unstable constructs held back as plain
 * text. Recomputed from scratch on every update, so transient guesses
 * self-correct as more tokens arrive.
 */
export function repairTail(tail: string): RepairedTail {
	const lines = splitLines(tail);

	// 1. Open code fence → close it; everything inside is literal.
	let fence: string | null = null;
	for (const l of lines) fence = fenceStep(l.text, fence);
	if (fence !== null) {
		const closer = fence[0].repeat(Math.max(3, fence.length));
		return {
			md: tail + (tail.endsWith('\n') ? closer : '\n' + closer),
			inlinePlain: '',
			blockPlain: '',
		};
	}

	// 2. Hold back a trailing not-yet-live table: a run of `|` lines at a
	//    block start that doesn't have a completed delimiter row yet.
	let md = tail;
	let blockPlain = '';
	let runStart = lines.length;
	while (runStart > 0 && /^ {0,3}\|/.test(lines[runStart - 1].text)) runStart--;
	if (runStart < lines.length) {
		const run = lines.slice(runStart);
		const atBlockStart =
			runStart === 0 || BLANK_RE.test(lines[runStart - 1].text);
		// Live once the delimiter row is complete (a later line or a trailing
		// newline exists after it) — from then on marked renders a real table
		// and partial body rows fill in cell by cell.
		const live =
			run.length >= 2 &&
			isDelimiterRow(run[1].text) &&
			(run.length >= 3 || tail.endsWith('\n'));
		if (atBlockStart && !live) {
			blockPlain = tail.slice(run[0].start);
			md = tail.slice(0, run[0].start);
		}
	}

	// 3. Dangling link/image on the last line: `[text` or `[text](partial-url`
	//    can't be stabilized — hold it back as plain text at the cursor.
	let inlinePlain = '';
	if (!blockPlain) {
		const m = /(!?\[[^\]\n]*(?:\]\([^()\n]*)?)$/.exec(md);
		if (m) {
			inlinePlain = m[1];
			md = md.slice(0, m.index);
		}
	}

	// 4. Close unbalanced inline constructs in the final block.
	md += inlineClosers(md);

	return { md, inlinePlain, blockPlain };
}

/**
 * Computes the closer suffix (e.g. "`" + "**") for inline constructs left
 * open in the *final block* of `md` — appended closers can only ever repair
 * the last block, and unclosed markers in earlier blocks already render as
 * literal text, which is stable.
 */
function inlineClosers(md: string): string {
	const lines = splitLines(md);

	// Find the start of the final block: after the last blank line, fence
	// close, or heading; list items / blockquote lines start their own block.
	let fence: string | null = null;
	let segFrom = 0;
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i].text;
		const before = fence;
		fence = fenceStep(line, fence);
		if (before !== null) {
			if (fence === null) segFrom = i + 1; // fence just closed
			continue;
		}
		if (fence !== null) continue; // fence just opened (line consumed)
		if (BLANK_RE.test(line) || HEADING_RE.test(line)) segFrom = i + 1;
		else if (LIST_ITEM_RE.test(line)) segFrom = i;
	}
	if (segFrom >= lines.length) return '';

	let codeTicks = 0;
	const stack: string[] = [];

	for (let li = segFrom; li < lines.length; li++) {
		// Strip block prefixes that aren't inline content.
		let line = lines[li].text.replace(/^ {0,3}(?:> ?)*/, '');
		if (li === segFrom) line = line.replace(LIST_ITEM_RE, '');

		for (let i = 0; i < line.length; i++) {
			const ch = line[i];
			if (ch === '\\') {
				i++; // escaped char
				continue;
			}
			if (ch === '`') {
				let n = 1;
				while (line[i + n] === '`') n++;
				if (codeTicks === 0) codeTicks = n;
				else if (n === codeTicks) codeTicks = 0;
				i += n - 1;
				continue;
			}
			if (codeTicks > 0) continue; // literal inside inline code

			if (ch === '~' && line[i + 1] === '~') {
				toggle(stack, '~~');
				i++;
				continue;
			}
			if (ch === '*' || ch === '_') {
				let n = 1;
				while (line[i + n] === ch) n++;
				const prev = i > 0 ? line[i - 1] : '';
				const next = line[i + n] ?? '';
				let rem = n;
				if (rem >= 2) {
					emphasis(stack, ch + ch, prev, next, ch);
					rem -= 2;
				}
				if (rem >= 1) emphasis(stack, ch, prev, next, ch);
				i += n - 1;
				continue;
			}
		}
	}

	let closers = codeTicks > 0 ? '`'.repeat(codeTicks) : '';
	for (let i = stack.length - 1; i >= 0; i--) closers += stack[i];
	return closers;
}

function toggle(stack: string[], marker: string): void {
	const at = stack.lastIndexOf(marker);
	if (at !== -1) stack.splice(at, 1);
	else stack.push(marker);
}

/**
 * Open/close heuristic for `*`/`**`/`_`/`__` runs. `prev`/`next` are the
 * characters around the whole run. Closes when the marker is open and the
 * run can close (non-space before); opens when the run can open (non-space
 * after; `_` additionally requires a word boundary before — no intra-word
 * `_` emphasis in GFM). Anything else is literal (e.g. `2 ** 3`).
 */
function emphasis(stack: string[], marker: string, prev: string, next: string, ch: string): void {
	const open = stack.lastIndexOf(marker) !== -1;
	const canClose = prev !== '' && !/\s/.test(prev);
	const wordBefore = /[\w]/.test(prev);
	const canOpen =
		next !== '' && !/\s/.test(next) && (ch === '*' || !wordBefore);
	if (open && canClose) stack.splice(stack.lastIndexOf(marker), 1);
	else if (canOpen) stack.push(marker);
}
