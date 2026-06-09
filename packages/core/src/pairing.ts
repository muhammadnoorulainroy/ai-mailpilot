/**
 * Pairing code mechanism that lets the extension securely fetch the Core auth token
 * once, with short-lived, single-use, attempt-limited codes to deter brute-force.
 */
import { randomInt } from 'node:crypto';
import type { Logger } from 'pino';

const CODE_TTL_MS = 15 * 60 * 1000;
const MAX_ATTEMPTS = 10;

/** Result of redeeming a pairing code, either the auth token or an error with an HTTP status. */
export type RedeemResult = { token: string } | { error: string; status: number };

/**
 * Short-lived, single-use, attempt-limited pairing code that lets the extension
 * fetch the auth token, preventing brute-force of the unauthenticated pair endpoint.
 */
export class Pairing {
  private readonly code: string;
  private readonly expiresAt: number;
  private attemptsLeft = MAX_ATTEMPTS;
  private used = false;

  /** Generate a random 6-digit code (or use the supplied one), set its expiry, and log it for the user. */
  constructor(
    private readonly token: string,
    logger: Logger,
    code?: string,
  ) {
    this.code = code ?? String(randomInt(0, 1_000_000)).padStart(6, '0');
    this.expiresAt = Date.now() + CODE_TTL_MS;
    logger.info(
      { pairingCode: this.code },
      `Pairing code: ${this.code} - enter it in the extension Settings to connect (valid ${CODE_TTL_MS / 60000} min, needed once).`,
    );
  }

  /** Exchange a code for the auth token, enforcing single-use, expiry and attempt limits. */
  redeem(code: string): RedeemResult {
    if (this.used)
      return { error: 'pairing already completed; restart Core to pair again', status: 410 };
    if (Date.now() > this.expiresAt)
      return { error: 'pairing code expired; restart Core for a new one', status: 410 };
    if (this.attemptsLeft <= 0)
      return { error: 'too many attempts; restart Core for a new code', status: 429 };
    if (code !== this.code) {
      this.attemptsLeft -= 1;
      return { error: 'incorrect pairing code', status: 401 };
    }
    this.used = true;
    return { token: this.token };
  }
}
