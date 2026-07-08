/**
 * User-owned mail organization captured from Thunderbird: the user's own tags and their meaningful
 * folder, per message. These are signals only, never MailPilot AI categories. MailPilot-managed tags
 * (mailpilot_ prefix) are never stored here. Writes use a per-message replace so a tag removed or a
 * message moved in Thunderbird is reflected on the next sync, without touching other messages.
 */
import type { Database, Statement } from 'better-sqlite3';

/** Where a user label came from. */
export type UserLabelSource = 'thunderbird_tag' | 'folder';

/** One user label to store for a message. */
export interface UserLabelInput {
  source: UserLabelSource;
  key: string;
  label: string;
}

/** A stored user label for a message. */
export interface EmailUserLabel {
  source: UserLabelSource;
  key: string;
  label: string;
}

/** An aggregated user label with how many of the account's messages carry it. */
export interface UserLabelStat {
  source: UserLabelSource;
  key: string;
  label: string;
  count: number;
}

/** Data access for user-owned Thunderbird tags and folder labels. */
export class EmailUserLabelRepository {
  private readonly stmts: {
    deleteForEmail: Statement<unknown[]>;
    insert: Statement<unknown[]>;
    forEmail: Statement<unknown[]>;
    stats: Statement<unknown[]>;
    countDistinct: Statement<unknown[]>;
    subjectsForLabel: Statement<unknown[]>;
    messageIdsForLabel: Statement<unknown[]>;
  };

  constructor(private db: Database) {
    this.stmts = {
      deleteForEmail: db.prepare(
        'DELETE FROM email_user_labels WHERE account_id = ? AND message_id = ?',
      ),
      insert: db.prepare(
        `INSERT INTO email_user_labels (account_id, message_id, source, key, label, synced_at)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT (account_id, message_id, source, key)
           DO UPDATE SET label = excluded.label, synced_at = excluded.synced_at`,
      ),
      forEmail: db.prepare(
        'SELECT source, key, label FROM email_user_labels WHERE account_id = ? AND message_id = ? ORDER BY source, label',
      ),
      stats: db.prepare(
        `SELECT source, key, MAX(label) AS label, COUNT(*) AS count
           FROM email_user_labels
          WHERE account_id = ? AND source = ?
          GROUP BY source, key
          ORDER BY count DESC, label ASC
          LIMIT ?`,
      ),
      countDistinct: db.prepare(
        'SELECT COUNT(DISTINCT key) AS n FROM email_user_labels WHERE account_id = ? AND source = ?',
      ),
      subjectsForLabel: db.prepare(
        `SELECT e.subject AS subject
           FROM email_user_labels l
           JOIN emails e ON e.message_id = l.message_id AND e.account_id = l.account_id
          WHERE l.account_id = ? AND l.source = ? AND l.key = ?
            AND e.subject IS NOT NULL AND e.subject != ''
          ORDER BY e.date DESC
          LIMIT ?`,
      ),
      messageIdsForLabel: db.prepare(
        `SELECT message_id AS messageId FROM email_user_labels
          WHERE account_id = ? AND source = ? AND key = ?
          LIMIT ?`,
      ),
    };
  }

  /**
   * Replace every stored user label for each given message with the supplied set, in one transaction.
   * Messages not in the batch are untouched, so a partial sync never drops another message's labels.
   */
  replaceForEmails(
    accountId: string,
    entries: Array<{ messageId: string; labels: UserLabelInput[] }>,
    syncedAt: number,
  ): void {
    const tx = this.db.transaction(() => {
      for (const entry of entries) {
        this.stmts.deleteForEmail.run(accountId, entry.messageId);
        for (const label of entry.labels) {
          if (label.key.length === 0 || label.label.length === 0) continue;
          this.stmts.insert.run(
            accountId,
            entry.messageId,
            label.source,
            label.key,
            label.label,
            syncedAt,
          );
        }
      }
    });
    tx();
  }

  /** Replace one message's user labels. Convenience wrapper over replaceForEmails. */
  replaceForEmail(
    accountId: string,
    messageId: string,
    labels: UserLabelInput[],
    syncedAt: number,
  ): void {
    this.replaceForEmails(accountId, [{ messageId, labels }], syncedAt);
  }

  /** All stored user labels for one message. */
  labelsForEmail(accountId: string, messageId: string): EmailUserLabel[] {
    return this.stmts.forEmail.all(accountId, messageId) as EmailUserLabel[];
  }

  /** The most common user labels of a source across the account, most-used first. */
  topLabels(accountId: string, source: UserLabelSource, limit: number): UserLabelStat[] {
    return (
      this.stmts.stats.all(accountId, source, limit) as Array<{
        source: UserLabelSource;
        key: string;
        label: string;
        count: number;
      }>
    ).map((r) => ({ source: r.source, key: r.key, label: r.label, count: r.count }));
  }

  /** How many distinct labels of a source the account has. */
  countDistinct(accountId: string, source: UserLabelSource): number {
    return (this.stmts.countDistinct.get(accountId, source) as { n: number }).n;
  }

  /** Representative recent subjects for one label, for hint summaries and suggestions. */
  representativeSubjects(
    accountId: string,
    source: UserLabelSource,
    key: string,
    limit: number,
  ): string[] {
    return (
      this.stmts.subjectsForLabel.all(accountId, source, key, limit) as Array<{ subject: string }>
    ).map((r) => r.subject);
  }

  /** Message ids carrying one label, for coherence scoring. Bounded by limit. */
  messageIdsForLabel(
    accountId: string,
    source: UserLabelSource,
    key: string,
    limit: number,
  ): string[] {
    return (
      this.stmts.messageIdsForLabel.all(accountId, source, key, limit) as Array<{
        messageId: string;
      }>
    ).map((r) => r.messageId);
  }
}
