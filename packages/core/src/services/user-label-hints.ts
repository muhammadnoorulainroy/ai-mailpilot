/**
 * Summarizes the user's own Thunderbird tags and meaningful folders into a compact, weak-hint block
 * for discovery prompts (Part D). Only well-used labels are included, with a couple of representative
 * subjects each, so discovery can be aware of the user's organization without copying it. Generic
 * folders and MailPilot-managed tags are already excluded upstream at ingest.
 */
import type {
  EmailUserLabelRepository,
  UserLabelSource,
} from '../repositories/email-user-label-repository.js';

/** Fewest emails a label needs before it is worth hinting. */
const MIN_COUNT = 5;
const MAX_PER_SOURCE = 6;
const SUBJECTS_PER_LABEL = 2;
const SUBJECT_MAX = 60;

/** A summarized hint block plus how many labels it covers (0 = nothing worth hinting). */
export interface UserLabelHint {
  text: string;
  labelCount: number;
}

/** One trimmed, single-line, length-capped subject for a hint. */
function tidySubject(subject: string): string {
  return `"${subject.replace(/\s+/g, ' ').trim().slice(0, SUBJECT_MAX)}"`;
}

/**
 * Build the weak-hint block from an account's top user labels, or an empty hint when none clear the
 * usage floor. The caller decides whether it is allowed to include this (privacy/cloud policy).
 */
export function summarizeUserLabels(
  repo: EmailUserLabelRepository,
  accountId: string,
): UserLabelHint {
  const sections: string[] = [];
  let labelCount = 0;

  const render = (source: UserLabelSource, heading: string): void => {
    const stats = repo
      .topLabels(accountId, source, MAX_PER_SOURCE)
      .filter((s) => s.count >= MIN_COUNT);
    if (stats.length === 0) return;
    const lines = stats.map((s) => {
      const subjects = repo
        .representativeSubjects(accountId, source, s.key, SUBJECTS_PER_LABEL)
        .map(tidySubject)
        .join(', ');
      labelCount += 1;
      return `- ${s.label} (${s.count} emails)${subjects ? `: ${subjects}` : ''}`;
    });
    sections.push(`${heading}:\n${lines.join('\n')}`);
  };

  render('thunderbird_tag', 'User tags');
  render('folder', 'User folders');
  if (sections.length === 0) return { text: '', labelCount: 0 };

  const text =
    'The user already organizes some mail with the tags and folders below. Treat these ONLY as weak hints:\n' +
    `${sections.join('\n')}\n` +
    'A topic MAY align with one of these, but do NOT create a topic just because a tag or folder exists, do NOT ' +
    'copy a personal or noisy label as a topic, and keep every topic purpose-based. These are the user’s own ' +
    'labels and stay separate from the AI categories you propose.';
  return { text, labelCount };
}
