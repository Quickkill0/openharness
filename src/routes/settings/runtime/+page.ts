// /settings/runtime merged into /settings/engines (M16 W6) — redirect kept
// so old links/bookmarks keep working.
import { redirect } from '@sveltejs/kit';

export function load(): never {
	redirect(308, '/settings/engines');
}
