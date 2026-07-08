/**
 * Import-prep only (Part E): proposes which user-owned Thunderbird tags/folders could seed AI
 * categories. It never creates or applies a category. A suggestion carries how many emails use the
 * label, a best-effort coherence score, representative subjects, and a cleaned suggested AI label so
 * the user can decide. Local, read-only, no LLM.
 */
import type {
  EmailUserLabelRepository,
  UserLabelSource,
} from '../repositories/email-user-label-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { cosineSimilarity, meanNormalize } from '../util/vector.js';

/** Fewest emails a label needs before it is worth proposing as a category seed. */
const MIN_COUNT = 4;
const MAX_SUGGESTIONS = 20;
/** Cap on embeddings loaded per label when scoring coherence. */
const COHERENCE_SAMPLE = 60;
const SUBJECT_SAMPLE = 3;

/** A user label proposed as a possible AI category seed. Never applied automatically. */
export interface UserLabelSuggestion {
  source: UserLabelSource;
  key: string;
  label: string;
  count: number;
  /** Mean cosine of the label's emails to their own centroid, or null when not computable. */
  coherence: number | null;
  representativeSubjects: string[];
  suggestedCategoryLabel: string;
}

/** Builds import-prep suggestions from the user's own Thunderbird organization. */
export class UserLabelSuggestionService {
  constructor(
    private userLabels: EmailUserLabelRepository,
    private embeddings: EmbeddingRepository,
  ) {}

  /**
   * Propose the account's most-used, meaningful user labels as category seeds. Coherence is scored
   * only when an embedding model is given and the label's emails are embedded.
   */
  suggest(accountId: string, embeddingModelId?: string): UserLabelSuggestion[] {
    const out: UserLabelSuggestion[] = [];
    for (const source of ['thunderbird_tag', 'folder'] as const) {
      for (const stat of this.userLabels.topLabels(accountId, source, MAX_SUGGESTIONS)) {
        if (stat.count < MIN_COUNT) continue;
        out.push({
          source,
          key: stat.key,
          label: stat.label,
          count: stat.count,
          coherence: embeddingModelId
            ? this.coherence(accountId, source, stat.key, embeddingModelId)
            : null,
          representativeSubjects: this.userLabels.representativeSubjects(
            accountId,
            source,
            stat.key,
            SUBJECT_SAMPLE,
          ),
          suggestedCategoryLabel: toTitleCase(stat.label),
        });
      }
    }
    return out.sort((a, b) => b.count - a.count).slice(0, MAX_SUGGESTIONS);
  }

  /** Mean cosine of a label's emails to their own centroid, or null when too few are embedded. */
  private coherence(
    accountId: string,
    source: UserLabelSource,
    key: string,
    modelId: string,
  ): number | null {
    const ids = this.userLabels.messageIdsForLabel(accountId, source, key, COHERENCE_SAMPLE);
    const vectors: Float32Array[] = [];
    for (const messageId of ids) {
      const vec = this.embeddings.getEmbedding({ messageId, accountId, modelId });
      if (vec) vectors.push(vec);
    }
    if (vectors.length < 2) return null;
    const mean = meanNormalize(vectors);
    if (!mean) return null;
    return vectors.reduce((sum, v) => sum + cosineSimilarity(v, mean), 0) / vectors.length;
  }
}

/** Trim, collapse whitespace, and title-case a user label into a candidate AI category label. */
function toTitleCase(label: string): string {
  return label
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
