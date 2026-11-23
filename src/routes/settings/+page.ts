// The settings hub is now a sub-nav; its first section is Engines.
import { redirect } from '@sveltejs/kit';

export function load(): never {
	redirect(307, '/settings/engines');
}
