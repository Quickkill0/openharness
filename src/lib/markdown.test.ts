import { describe, expect, it } from 'vitest';
import { marked } from 'marked';
import { findStableBoundary, repairTail } from './markdown';

describe('findStableBoundary', () => {
	it('returns 0 when there is no blank line yet', () => {
		expect(findStableBoundary('streaming a single paragraph')).toBe(0);
	});

	it('splits after a blank line between two paragraphs', () => {
		const text = 'first paragraph\n\nsecond paragraph';
		const b = findStableBoundary(text);
		expect(text.slice(0, b)).toBe('first paragraph\n\n');
		expect(text.slice(b)).toBe('second paragraph');
	});

	it('returns the last qualifying boundary', () => {
		const text = 'one\n\ntwo\n\nthree is still streaming';
		const b = findStableBoundary(text);
		expect(text.slice(b)).toBe('three is still streaming');
	});

	it('ignores blank lines inside an open code fence', () => {
		expect(findStableBoundary('```\ncode\n\nmore code')).toBe(0);
	});

	it('splits after a closed code fence', () => {
		const text = 'intro\n\n```\ncode\n```\n\nafter the fence';
		const b = findStableBoundary(text);
		expect(text.slice(b)).toBe('after the fence');
	});

	it('does not split between two list items (loose-list merge)', () => {
		expect(findStableBoundary('- item one\n\n- item two')).toBe(0);
	});

	it('does not split inside an indented code block', () => {
		expect(findStableBoundary('    code line\n\n    more code')).toBe(0);
	});

	it('splits between a paragraph and a following list', () => {
		const text = 'intro paragraph\n\n- first item';
		const b = findStableBoundary(text);
		expect(text.slice(b)).toBe('- first item');
	});

	it('does not treat a trailing blank line as a boundary (next block not started)', () => {
		expect(findStableBoundary('paragraph\n\n')).toBe(0);
	});
});

describe('findStableBoundary — split-parse equivalence', () => {
	// The streaming component parses prefix and tail separately and caches the
	// prefix HTML; at a stable boundary that must equal parsing the whole text.
	const cases: [string, string][] = [
		['paragraphs', 'first paragraph\n\nsecond one with **bold** text'],
		['heading then list', '## Title\n\n- one\n- two\n\nclosing paragraph'],
		['closed fence then prose', 'intro\n\n```js\nconst x = 1;\n```\n\nafter'],
		['blockquotes', '> quoted\n\nplain follow-up'],
	];
	for (const [name, text] of cases) {
		it(`prefix+tail HTML equals whole-document HTML (${name})`, () => {
			const b = findStableBoundary(text);
			expect(b).toBeGreaterThan(0);
			const split =
				(marked(text.slice(0, b)) as string) + (marked(text.slice(b)) as string);
			expect(split).toBe(marked(text) as string);
		});
	}
});

describe('repairTail — code fences', () => {
	it('closes an unclosed fence so it renders as a code block immediately', () => {
		const t = repairTail('```js\nconst x = 1');
		expect(t.md).toBe('```js\nconst x = 1\n```');
		expect(t.inlinePlain).toBe('');
		expect(t.blockPlain).toBe('');
	});

	it('closes an unclosed fence that ends with a newline without doubling it', () => {
		expect(repairTail('```\nfoo\n').md).toBe('```\nfoo\n```');
	});

	it('matches the opening fence length when closing', () => {
		expect(repairTail('````\ncode with ``` inside').md).toBe(
			'````\ncode with ``` inside\n````',
		);
	});

	it('leaves a closed fence untouched', () => {
		const md = '```\ndone\n```';
		expect(repairTail(md).md).toBe(md);
	});

	it('does not apply inline repairs to fence content', () => {
		// `**` inside the fence is literal — only the fence closer is added.
		expect(repairTail('```\na ** b').md).toBe('```\na ** b\n```');
	});

	it('repairs inline constructs after a closed fence', () => {
		expect(repairTail('```\ncode\n```\n**bold').md).toBe('```\ncode\n```\n**bold**');
	});
});

describe('repairTail — inline constructs', () => {
	it('passes plain text through unchanged', () => {
		const t = repairTail('just plain words');
		expect(t).toEqual({ md: 'just plain words', inlinePlain: '', blockPlain: '' });
	});

	it('closes unclosed inline code', () => {
		expect(repairTail('use `pnpm check').md).toBe('use `pnpm check`');
	});

	it('leaves balanced inline code untouched', () => {
		expect(repairTail('run `pnpm check` now').md).toBe('run `pnpm check` now');
	});

	it('closes unclosed bold', () => {
		expect(repairTail('this is **important').md).toBe('this is **important**');
	});

	it('closes unclosed italic', () => {
		expect(repairTail('an *emphasis').md).toBe('an *emphasis*');
	});

	it('closes bold-italic in the right order', () => {
		expect(repairTail('***both').md).toBe('***both***');
	});

	it('closes unclosed underscore emphasis', () => {
		expect(repairTail('some _emphasis').md).toBe('some _emphasis_');
		expect(repairTail('very __strong').md).toBe('very __strong__');
	});

	it('does not treat intra-word underscores as emphasis', () => {
		expect(repairTail('call snake_case_fn now').md).toBe('call snake_case_fn now');
	});

	it('closes unclosed strikethrough', () => {
		expect(repairTail('~~gone').md).toBe('~~gone~~');
	});

	it('does not mistake a list bullet for emphasis', () => {
		expect(repairTail('* item one').md).toBe('* item one');
	});

	it('closes bold inside the last list item', () => {
		expect(repairTail('- plain\n- has **bold').md).toBe('- plain\n- has **bold**');
	});

	it('treats spaced asterisks as literal (2 ** 3)', () => {
		expect(repairTail('compute 2 ** 3').md).toBe('compute 2 ** 3');
	});

	it('treats asterisks inside inline code as literal', () => {
		expect(repairTail('`a * b').md).toBe('`a * b`');
	});

	it('closes inline code before the enclosing bold', () => {
		expect(repairTail('**bold `code').md).toBe('**bold `code`**');
	});

	it('only repairs the final block — earlier unbalanced markers stay literal', () => {
		// The `**` in the first paragraph renders literally either way; the
		// appended closer must not bold the second paragraph.
		expect(repairTail('odd ** marker\n\nnext paragraph').md).toBe(
			'odd ** marker\n\nnext paragraph',
		);
	});

	it('does not let an unclosed marker in a heading leak into following text', () => {
		expect(repairTail('## **head\nbody text').md).toBe('## **head\nbody text');
	});

	it('skips escaped markers', () => {
		expect(repairTail('a \\*literal star').md).toBe('a \\*literal star');
	});
});

describe('repairTail — held-back unstable constructs', () => {
	it('holds back a table header until its delimiter row completes', () => {
		const t = repairTail('| name | size |\n| ---');
		expect(t.md).toBe('');
		expect(t.blockPlain).toBe('| name | size |\n| ---');
	});

	it('holds back a lone table-row start', () => {
		const t = repairTail('intro:\n\n| name ');
		expect(t.md).toBe('intro:\n\n');
		expect(t.blockPlain).toBe('| name ');
	});

	it('renders a live table once the delimiter row is complete', () => {
		const t = repairTail('| a | b |\n| --- | --- |\n| 1 | 2');
		expect(t.md).toBe('| a | b |\n| --- | --- |\n| 1 | 2');
		expect(t.blockPlain).toBe('');
	});

	it('holds back a dangling link label', () => {
		const t = repairTail('see [the docs');
		expect(t.md).toBe('see ');
		expect(t.inlinePlain).toBe('[the docs');
	});

	it('holds back a link with a partial url', () => {
		const t = repairTail('see [docs](https://exa');
		expect(t.md).toBe('see ');
		expect(t.inlinePlain).toBe('[docs](https://exa');
	});

	it('holds back a dangling image', () => {
		const t = repairTail('shot: ![alt text');
		expect(t.md).toBe('shot: ');
		expect(t.inlinePlain).toBe('![alt text');
	});

	it('leaves a completed link alone', () => {
		const md = 'see [docs](https://example.com) for more';
		expect(repairTail(md)).toEqual({ md, inlinePlain: '', blockPlain: '' });
	});
});
