// /settings/vram ("GPU & Models") merged into Settings, Hardware in the
// Workbench redesign; redirect kept so old links keep working.
import { redirect } from '@sveltejs/kit';

export function load(): never {
	redirect(308, '/settings/hardware');
}
