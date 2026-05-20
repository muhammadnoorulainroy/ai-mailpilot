/**
 * Helpers for normalizing model identifiers so equivalent forms map consistently.
 */

/**
 * Strip `:latest` so a model and its tagged form map to the same storage row.
 * Keeps embeddings, centroids, and triage results aligned across surface mismatches.
 */
export function canonicalizeModelId(id: string): string {
  return id.trim().replace(/:latest$/i, '');
}
